// Host-RAM prefix reuse substrate contracts on the quasar-master baseline (V2-T8).
//
// Verifies the observable behavior of detail::KVRamCache as ported onto the baseline's
// mechanisms (DeviceKVPagePool + shared HostKVArena for the paged image, complete
// StateImage payloads for the state half):
//   * paged-KV image round trip (device -> host arena -> device), including captures
//     with non-contiguous physical page runs
//   * complete StateImage round trip (linear conv/recurrent, continuation hidden,
//     DFlash local cyclic K/V)
//   * hash-chain index: honest prefix_hash_chain, longest-match, frontier beats
//     checkpoint, checkpoint fallback, exclusive-claim hiding, consume-erase,
//     multi-claim stays matchable
//   * tiered FIFO eviction: a lineage-demonstrated (protected) record outlives cold
//     records under capacity pressure
//   * destructor safety with in-flight capture copies, and arena reuse afterwards
//
// CUDA tests: returns 77 (SKIP) when no usable device is present, like the other
// CUDA test targets in this tree.

#include "core/arena.h"
#include "core/device.h"
#include "core/host_kv_arena.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/state_image.h>

#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <cuda_runtime.h>
#include <algorithm>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;
namespace detail = q36::detail;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

// ---------------------------------------------------------------- fixtures

// One decoder-state backing (owns its text KV page pool). Members are initialized in
// declaration order so the arena exists before the state references its bytes.
struct KvState {
    explicit KvState(std::size_t bytes, const q36::DecoderStateLayout& layout)
        : arena(bytes), state({arena.base(), arena.capacity()}, layout) {}

    ninfer::DeviceArena arena;
    q36::DecoderState state;
};

KvState plan_kv_state(std::uint32_t page_groups) {
    q36::DecoderStateSpec spec{
        .full_attention_layers     = 2,
        .capacity                  = 129,
        .kv_heads                  = 2,
        .attention_head_dim        = 256,
        .kv_storage                = ninfer::KvCacheStorage::Nvfp4Group16,
        .text_physical_page_groups = page_groups,
    };
    ninfer::LayoutBuilder builder;
    const q36::DecoderStateLayout layout = q36::plan_decoder_state(builder, spec);
    return KvState(builder.finish(256), layout);
}

struct StatePools {
    explicit StatePools(std::size_t bytes, const q36::StateImageDeviceLayout& layout)
        : arena(bytes), pool({arena.base(), arena.capacity()}, layout) {}

    ninfer::DeviceArena arena;
    q36::StateImageDevicePool pool;
};

StatePools plan_state_pools(std::int32_t slots) {
    q36::StateImageSpec spec{
        .linear =
            {
                .layers         = 2,
                .conv_channels  = 5,
                .conv_width     = 3,
                .value_heads    = 2,
                .value_head_dim = 4,
                .key_head_dim   = 3,
                .slot_count     = slots,
                .conv_dtype     = ninfer::DType::BF16,
            },
        .hidden = 7,
    };
    spec.dflash_local = q36::DFlashLocalStateSpec{.layers = 2, .capacity = 17, .kv_heads = 2,
                                                  .head_dim = 4};
    ninfer::LayoutBuilder builder;
    const q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return StatePools(builder.finish(256), layout);
}

std::vector<ninfer::DeviceKVPageHandle> handles(std::span<const ninfer::DeviceKVPageLease> leases) {
    std::vector<ninfer::DeviceKVPageHandle> out;
    out.reserve(leases.size());
    for (const auto& lease : leases) { out.push_back(lease.handle()); }
    return out;
}

// A fresh pool materializes from the free-list head, so lease i owns physical page i.
std::vector<ninfer::DeviceKVPageLease>
materialize(q36::PagedKVCache& kv, std::uint32_t pages) {
    auto reservation = kv.page_pool().reserve(pages);
    if (!reservation.has_value()) {
        std::cerr << "FAIL: KV reservation failed\n";
        std::exit(1);
    }
    std::vector<ninfer::DeviceKVPageLease> leases;
    leases.reserve(pages);
    kv.page_pool().materialize(*reservation, pages, leases);
    return leases;
}

// ---------------------------------------------------------------- KV fill/check

// Pattern per lease position i (NOT per physical page): seed + i*17 + 1, written into
// the physical page the lease owns. physical_ids[i] is that page's physical index.
void fill_pages(const q36::PagedKVCache& kv, std::span<const std::int32_t> physical_ids,
                std::size_t count, unsigned char seed, const char* label) {
    const auto& pool = kv.page_pool();
    const auto order = pool.geometry().device_plane_order;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& t = pool.plane(plane);
        std::vector<unsigned char> host(t.bytes(), 0);
        CUDA_CHECK(cudaMemcpy(host.data(), t.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned char value = static_cast<unsigned char>(seed + i * 17U + 1U);
            const std::int32_t page   = physical_ids[i];
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(page) * static_cast<std::size_t>(t.nb[3]);
                std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                          host.begin() + static_cast<std::ptrdiff_t>(begin + t.nb[3]), value);
            } else {
                const std::size_t stride = static_cast<std::size_t>(t.nb[3]);
                for (std::int32_t head = 0; head < t.ne[3]; ++head) {
                    const std::size_t begin =
                        static_cast<std::size_t>(head) * stride +
                        static_cast<std::size_t>(page) * static_cast<std::size_t>(t.nb[2]);
                    std::fill(host.begin() + static_cast<std::ptrdiff_t>(begin),
                              host.begin() + static_cast<std::ptrdiff_t>(begin + t.nb[2]),
                              static_cast<unsigned char>(value + static_cast<unsigned>(head)));
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(t.data, host.data(), host.size(), cudaMemcpyHostToDevice));
    }
}

int check_pages(const q36::PagedKVCache& kv, std::span<const std::int32_t> physical_ids,
                std::size_t count, unsigned char seed, std::string_view label) {
    const auto& pool = kv.page_pool();
    const auto order = pool.geometry().device_plane_order;
    int bad = 0;
    for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
        const ninfer::Tensor& t = pool.plane(plane);
        std::vector<unsigned char> host(t.bytes());
        CUDA_CHECK(cudaMemcpy(host.data(), t.data, host.size(), cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i < count; ++i) {
            const unsigned char value = static_cast<unsigned char>(seed + i * 17U + 1U);
            const std::int32_t page   = physical_ids[i];
            if (order == ninfer::PagedKVPlaneOrder::PageMajor) {
                const std::size_t begin = static_cast<std::size_t>(page) * static_cast<std::size_t>(t.nb[3]);
                for (std::int64_t byte = 0; byte < t.nb[3]; ++byte) {
                    if (host[begin + static_cast<std::size_t>(byte)] != value) {
                        if (bad == 0) { std::cerr << label << " page-major byte mismatch\n"; }
                        ++bad;
                        break;
                    }
                }
            } else {
                for (std::int32_t head = 0; head < t.ne[3]; ++head) {
                    const unsigned char expected =
                        static_cast<unsigned char>(value + static_cast<unsigned>(head));
                    const std::size_t begin =
                        static_cast<std::size_t>(head) * static_cast<std::size_t>(t.nb[3]) +
                        static_cast<std::size_t>(page) * static_cast<std::size_t>(t.nb[2]);
                    for (std::int64_t byte = 0; byte < t.nb[2]; ++byte) {
                        if (host[begin + static_cast<std::size_t>(byte)] != expected) {
                            if (bad == 0) { std::cerr << label << " head-major byte mismatch\n"; }
                            ++bad;
                            break;
                        }
                    }
                }
            }
        }
    }
    return bad;
}

// ---------------------------------------------------------------- state fill/check

void set_bytes(const ninfer::Tensor& tensor, unsigned char value) {
    CUDA_CHECK(cudaMemset(tensor.data, value, tensor.bytes()));
}

void expect_bytes(const ninfer::Tensor& tensor, unsigned char expected, std::string_view label) {
    std::vector<unsigned char> host(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(host.data(), tensor.data, host.size(), cudaMemcpyDeviceToHost));
    for (const unsigned char value : host) {
        if (value != expected) {
            expect(false, std::string(label) + " differs");
            return;
        }
    }
}

void fill_state_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        set_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer));
        set_bytes(pool.linear().recurrent_slot(layer, slot),
                  static_cast<unsigned char>(base + 0x10 + layer));
    }
    set_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20));
    if (auto* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            set_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer));
            set_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer));
        }
    }
}

void expect_state_slot(q36::StateImageDevicePool& pool, std::int32_t slot, unsigned char base,
                       std::string_view label) {
    for (std::uint32_t layer = 0; layer < pool.linear().layer_count(); ++layer) {
        expect_bytes(pool.linear().conv_slot(layer, slot), static_cast<unsigned char>(base + layer),
                     std::string(label) + " conv");
        expect_bytes(pool.linear().recurrent_slot(layer, slot),
                     static_cast<unsigned char>(base + 0x10 + layer), std::string(label) + " recurrent");
    }
    expect_bytes(pool.continuation_hidden_slot(slot), static_cast<unsigned char>(base + 0x20),
                 std::string(label) + " hidden");
    if (auto* local = pool.dflash_local(); local != nullptr) {
        for (std::uint32_t layer = 0; layer < local->layer_count(); ++layer) {
            const auto view = local->layer_view(layer);
            expect_bytes(view.k.slice(3, slot, 1), static_cast<unsigned char>(base + 0x30 + layer),
                         std::string(label) + " dflash K");
            expect_bytes(view.v.slice(3, slot, 1), static_cast<unsigned char>(base + 0x40 + layer),
                         std::string(label) + " dflash V");
        }
    }
}

// ---------------------------------------------------------------- capture plumbing

q36::PreparedPromptData text_prompt(std::vector<ninfer::TokenId> tokens) {
    q36::PreparedPromptData prompt;
    prompt.token_ids   = std::move(tokens);
    prompt.token_types.assign(prompt.token_ids.size(), 0);
    prompt.positions.resize(3 * prompt.token_ids.size());
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < prompt.token_ids.size(); ++i) {
            prompt.positions[static_cast<std::size_t>(axis) * prompt.token_ids.size() + i] =
                static_cast<std::int32_t>(i);
        }
    }
    return prompt;
}

// The retained form of a prompt: the prompt plus one trailing token (what a lane's ledger
// holds after the request tokens).
q36::PreparedPromptData retained_prompt(const q36::PreparedPromptData& prompt) {
    q36::PreparedPromptData retained = prompt;
    retained.token_ids.push_back(0);
    retained.token_types.push_back(0);
    const std::size_t tokens = retained.token_ids.size();
    retained.positions.resize(3 * tokens);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::size_t i = 0; i < tokens; ++i) {
            retained.positions[static_cast<std::size_t>(axis) * tokens + i] =
                static_cast<std::int32_t>(i);
        }
    }
    return retained;
}

detail::RamCaptureSource
make_capture_source(q36::PagedKVCache& kv, std::span<const ninfer::DeviceKVPageLease> leases,
                    const q36::PreparedPromptData& retained, std::uint32_t execution_frontier,
                    cudaStream_t stream, detail::ResidentPrefixIdentity& identity,
                    bool multi_claim = false,
                    std::uint32_t checkpoint_frontier = 0) {
    identity.assign(retained);
    detail::RamCaptureSource source;
    source.execution_frontier = execution_frontier;
    source.ledger_frontier    = execution_frontier + 1;
    source.text_kv_valid      = static_cast<std::uint32_t>(leases.size());
    source.ledger             = retained.token_ids;
    source.identity           = &identity;
    source.hash_f             = detail::prefix_hash_at(retained.token_ids, identity,
                                                       execution_frontier);
    source.text_cache          = &kv;
    source.stream              = stream;
    source.capture_kind        = detail::RamCaptureKind::Terminal;
    source.multi_claim         = multi_claim;
    if (checkpoint_frontier != 0) {
        source.rewrite_valid     = true;
        source.rewrite_kind      = q36::RewriteCheckpointKind::TurnClosure;
        source.rewrite_frontier  = checkpoint_frontier;
        source.hash_c_valid      = true;
        source.hash_c             = detail::prefix_hash_at(retained.token_ids, identity,
                                                            checkpoint_frontier);
    }
    return source;
}

struct CaptureSourceRef {
    std::vector<ninfer::DeviceKVPageHandle> text_handles;
    detail::RamCaptureSource source{};

    // Baseline Ram*Source/Target page spans are non-owning views (the engine keeps its lease
    // vectors); this wrapper owns the handle vector for the span's lifetime.
    CaptureSourceRef(q36::PagedKVCache& kv,
                     std::span<const ninfer::DeviceKVPageLease> leases,
                     const q36::PreparedPromptData& retained,
                     std::uint32_t execution_frontier,
                     cudaStream_t stream,
                     detail::ResidentPrefixIdentity& identity,
                     bool multi_claim = false,
                     std::uint32_t checkpoint_frontier = 0)
        : text_handles(handles(leases)) {
        source = make_capture_source(kv, leases, retained, execution_frontier, stream,
                                     identity, multi_claim, checkpoint_frontier);
        source.text_pages = text_handles;
    }
};

struct CaptureTarget {
    CaptureSourceRef source;
    q36::PreparedPromptData retained;
};

int capture_and_match(detail::KVRamCache& cache, const CaptureTarget& target,
                      const char* label) {
    if (!cache.capture(target.source.source)) {
        std::cerr << label << " capture failed\n";
        return 0;
    }
    auto match = cache.plan_match(target.retained, detail::prefix_hash_chain(target.retained));
    if (!match || match->reuse_base != target.source.source.execution_frontier ||
        match->reuse != ninfer::PrefixReusePath::AppendAtFrontier) {
        std::cerr << label << " capture did not match at its frontier\n";
        return 0;
    }
    return static_cast<int>(match->entry_id);
}

// ---------------------------------------------------------------- tests

int test_kv_roundtrip(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    KvState dst = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    auto source_leases = materialize(src.state.text_kv, 3);
    const std::vector<std::int32_t> source_ids{0, 1, 2};
    fill_pages(src.state.text_kv, source_ids, 3, 0x21, "kv roundtrip fill");
    ctx.synchronize();

    const auto prompt   = text_prompt({10, 100, 101, 102});
    const auto retained = retained_prompt(prompt);
    detail::ResidentPrefixIdentity identity;
    const CaptureTarget target{
        .source   = CaptureSourceRef(src.state.text_kv, source_leases, retained, 4, ctx.stream,
                                     identity),
        .retained = retained,
    };
    const int entry = capture_and_match(cache, target, "kv roundtrip");
    expect(entry != 0, "kv roundtrip capture indexed");
    if (entry == 0) { return bad; }

    auto destination_leases = materialize(dst.state.text_kv, 3);
    const std::vector<std::int32_t> destination_ids{0, 1, 2};
    cache.claim(entry);
    auto destination_handles = handles(destination_leases);
    detail::RamRestoreTarget restore{
        .text_pages = destination_handles,
        .text_cache = &dst.state.text_kv,
        .stream     = ctx.stream,
    };
    const detail::RamRestoredHost restored = cache.unpack_device(entry, restore);
    ctx.synchronize();
    expect(restored.execution_frontier == 4, "kv roundtrip restored frontier");
    expect(restored.ledger == retained.token_ids, "kv roundtrip restored ledger");
    bad += check_pages(dst.state.text_kv, destination_ids, 3, 0x21, "kv roundtrip restore");
    bad += check_pages(src.state.text_kv, source_ids, 3, 0x21, "kv roundtrip source intact");
    cache.consume(entry);
    expect(cache.plan_match(retained, detail::prefix_hash_chain(retained)) == std::nullopt,
           "kv roundtrip consumed exclusive entry is gone");
    return bad;
}

int test_irregular_runs(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(8);
    KvState dst = plan_kv_state(8);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    // Fragment the free list: own 5 (0..4), release physical 2, then materialize 4.
    // The baseline materialize(count) takes a single run of >= count when one exists
    // (releasing page 2 leaves runs [2,1) [5,3), so 3 pages would come back as [5,6,7]);
    // requesting 4 forces the stitch path: page 2 from the first run, 5,6,7 from the
    // second -> a two-run capture set.
    auto first = materialize(src.state.text_kv, 5);
    const bool released = first[2].release();
    expect(released, "irregular runs middle release");
    auto second = materialize(src.state.text_kv, 4);
    expect(src.state.text_kv.page_pool().contiguous_run_count(handles(second)) == 2,
           "irregular runs capture set is two runs");
    const std::vector<std::int32_t> source_ids{2, 5, 6, 7};
    fill_pages(src.state.text_kv, source_ids, 4, 0x45, "irregular runs fill");
    ctx.synchronize();

    const auto prompt   = text_prompt({10, 200, 201, 202});
    const auto retained = retained_prompt(prompt);
    detail::ResidentPrefixIdentity identity;
    const CaptureTarget target{
        .source  = CaptureSourceRef(src.state.text_kv, second, retained, 4, ctx.stream,
                                   identity),
        .retained = retained,
    };
    const int entry = capture_and_match(cache, target, "irregular runs");
    expect(entry != 0, "irregular runs capture indexed");
    if (entry == 0) { return bad; }

    auto destination_leases = materialize(dst.state.text_kv, 4);
    const std::vector<std::int32_t> destination_ids{0, 1, 2, 3};
    cache.claim(entry);
    auto destination_handles = handles(destination_leases);
    detail::RamRestoreTarget restore{
        .text_pages = destination_handles,
        .text_cache = &dst.state.text_kv,
        .stream     = ctx.stream,
    };
    cache.unpack_device(entry, restore);
    ctx.synchronize();
    bad += check_pages(dst.state.text_kv, destination_ids, 4, 0x45, "irregular runs restore");
    cache.consume(entry);
    return bad;
}

int test_state_image_roundtrip(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src_kv = plan_kv_state(4);
    KvState dst_kv = plan_kv_state(4);
    StatePools src_state = plan_state_pools(2);
    StatePools dst_state = plan_state_pools(2);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src_kv.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    auto source_leases = materialize(src_kv.state.text_kv, 2);
    const std::vector<std::int32_t> source_ids{0, 1};
    fill_pages(src_kv.state.text_kv, source_ids, 2, 0x67, "state roundtrip fill");
    fill_state_slot(src_state.pool, 0, 0x19);
    ctx.synchronize();

    const auto prompt   = text_prompt({10, 300});
    const auto retained = retained_prompt(prompt);
    detail::ResidentPrefixIdentity identity;
    CaptureSourceRef source(src_kv.state.text_kv, source_leases, retained, 2, ctx.stream,
                            identity);
    source.source.state_image = &src_state.pool;
    source.source.state_slot  = 0;
    const CaptureTarget target{.source = source, .retained = retained};
    const int entry = capture_and_match(cache, target, "state image roundtrip");
    expect(entry != 0, "state image roundtrip capture indexed");
    if (entry == 0) { return bad; }

    auto destination_leases = materialize(dst_kv.state.text_kv, 2);
    const std::vector<std::int32_t> destination_ids{0, 1};
    cache.claim(entry);
    auto destination_handles = handles(destination_leases);
    detail::RamRestoreTarget restore{
        .text_pages   = destination_handles,
        .text_cache   = &dst_kv.state.text_kv,
        .state_image  = &dst_state.pool,
        .state_slot   = 1,
        .stream       = ctx.stream,
    };
    cache.unpack_device(entry, restore);
    ctx.synchronize();
    expect_state_slot(dst_state.pool, 1, 0x19, "state image roundtrip restore");
    expect_state_slot(src_state.pool, 0, 0x19, "state image roundtrip source intact");
    bad += check_pages(dst_kv.state.text_kv, destination_ids, 2, 0x67,
                       "state image roundtrip kv restore");
    cache.consume(entry);
    return bad;
}

int test_index_match(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    auto leases = materialize(src.state.text_kv, 3);
    const std::vector<std::int32_t> ids{0, 1, 2};
    fill_pages(src.state.text_kv, ids, 3, 0x89, "index match fill");
    ctx.synchronize();

    const auto prompt_a    = text_prompt({10, 100, 101, 102, 103, 104});
    const auto retained_a  = retained_prompt(prompt_a);
    const auto prompt_b    = text_prompt({10, 100, 101, 202, 203});
    const auto retained_b  = retained_prompt(prompt_b);
    const auto prompt_c    = text_prompt({10, 100, 101, 301, 302, 303});
    const auto retained_c  = retained_prompt(prompt_c);

    detail::ResidentPrefixIdentity identity;
    const int entry_b = capture_and_match(
        cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_b, 5,
                                              ctx.stream, identity),
                .retained = retained_b},
        "index match B");
    const int entry_a = capture_and_match(
        cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_a, 6,
                                              ctx.stream, identity),
                .retained = retained_a},
        "index match A");
    const int entry_c = capture_and_match(
        cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_c, 6,
                                              ctx.stream, identity, true),
                .retained = retained_c},
        "index match C");
    expect(entry_a != 0 && entry_b != 0 && entry_c != 0, "index match all three indexed");
    if (entry_a == 0 || entry_b == 0 || entry_c == 0) { return bad; }

    // Shorter prompt: only its own entry matches (A's frontier is beyond the chain).
    auto match_b = cache.plan_match(retained_b, detail::prefix_hash_chain(retained_b));
    expect(match_b && match_b->entry_id == entry_b && match_b->reuse_base == 5,
           "index match shorter prompt finds its own entry");
    // Longer prompt: its own entry wins; B's frontier hash does not match A's chain.
    auto match_a = cache.plan_match(retained_a, detail::prefix_hash_chain(retained_a));
    expect(match_a && match_a->entry_id == entry_a && match_a->reuse_base == 6,
           "index match longer prompt finds its own entry");

    // Exclusive claim hides an entry from concurrent matchers; consume erases it.
    cache.claim(entry_b);
    expect(cache.plan_match(retained_b, detail::prefix_hash_chain(retained_b)) == std::nullopt,
           "claimed exclusive entry is hidden");
    auto restore_handles = handles(leases);
    detail::RamRestoreTarget restore{
        .text_pages = restore_handles,
        .text_cache = &src.state.text_kv,
        .stream     = ctx.stream,
    };
    cache.unpack_device(entry_b, restore);
    cache.consume(entry_b);
    expect(cache.plan_match(retained_b, detail::prefix_hash_chain(retained_b)) == std::nullopt,
           "consumed exclusive entry is gone");

    // Multi-claim: stays matchable while claimed and survives its own consume.
    cache.claim(entry_c);
    auto match_c = cache.plan_match(retained_c, detail::prefix_hash_chain(retained_c));
    expect(match_c && match_c->entry_id == entry_c && match_c->reuse_base == 6,
           "multi-claim entry stays matchable while claimed");
    cache.claim(entry_c);
    cache.unpack_device(entry_c, restore);
    cache.consume(entry_c);
    match_c = cache.plan_match(retained_c, detail::prefix_hash_chain(retained_c));
    expect(match_c && match_c->entry_id == entry_c, "multi-claim entry survives consume");
    cache.release(entry_c);
    return bad;
}

int test_checkpoint_fallback(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    auto leases = materialize(src.state.text_kv, 3);
    const std::vector<std::int32_t> ids{0, 1, 2};
    fill_pages(src.state.text_kv, ids, 3, 0xAB, "checkpoint fill");
    ctx.synchronize();

    const auto prompt = text_prompt({10, 100, 101, 102, 103, 104});
    const auto retained = retained_prompt(prompt);
    detail::ResidentPrefixIdentity identity;
    const int entry = capture_and_match(
        cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained, 6, ctx.stream,
                                              identity, false, /*checkpoint_frontier=*/3),
                .retained = retained},
        "checkpoint capture");
    expect(entry != 0, "checkpoint capture indexed");
    if (entry == 0) { return bad; }

    // A four-token prompt is too short for the frontier (6) but covers the checkpoint (3).
    const auto short_prompt   = text_prompt({10, 100, 101, 404});
    const auto short_retained = retained_prompt(short_prompt);
    auto match = cache.plan_match(short_retained, detail::prefix_hash_chain(short_retained));
    expect(match && match->entry_id == entry && match->reuse_base == 3 &&
               match->reuse == ninfer::PrefixReusePath::RestoreTurnCheckpoint,
           "checkpoint fallback for short prompt");
    // The full prompt prefers the frontier over the checkpoint.
    match = cache.plan_match(retained, detail::prefix_hash_chain(retained));
    expect(match && match->entry_id == entry && match->reuse_base == 6 &&
               match->reuse == ninfer::PrefixReusePath::AppendAtFrontier,
           "frontier beats checkpoint");
    return bad;
}

int test_hash_ladder(ninfer::DeviceContext& ctx) {
    int bad = 0;
    // Shape: sorted unique, covers [1, frontier], contains the frontier, every 2^j <= F,
    // and every F - 2^j >= 1.
    for (std::uint32_t frontier : {1u, 2u, 3u, 5u, 6u, 7u, 8u, 16u, 17u, 1000u}) {
        const auto points = detail::hash_ladder_points(frontier);
        expect(!points.empty() && points.front() >= 1 && points.back() == frontier,
               "ladder spans [1, frontier] and ends at the frontier");
        bool sorted_unique = true;
        for (std::size_t j = 1; j < points.size(); ++j) {
            sorted_unique = sorted_unique && points[j] > points[j - 1];
        }
        expect(sorted_unique, "ladder is sorted unique");
        expect(std::find(points.begin(), points.end(), frontier) != points.end(),
               "ladder contains the frontier");
        for (std::uint64_t power = 1; power <= frontier; power <<= 1) {
            expect(std::find(points.begin(), points.end(), static_cast<std::uint32_t>(power)) !=
                       points.end(),
                   "ladder contains 2^j");
            if (frontier - static_cast<std::uint32_t>(power) >= 1) {
                expect(std::find(points.begin(), points.end(),
                                 frontier - static_cast<std::uint32_t>(power)) != points.end(),
                       "ladder contains frontier - 2^j");
            }
        }
    }
    expect(detail::hash_ladder_points(0).empty(), "empty ladder for zero frontier");

    // Consistency: the single-pass ladder samples must equal both the per-point
    // prefix_hash_at and the full chain at the same frontiers.
    const auto prompt = text_prompt({7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43});
    const std::uint32_t frontier = static_cast<std::uint32_t>(prompt.token_ids.size());
    detail::ResidentPrefixIdentity identity;
    identity.assign(prompt);
    const auto chain = detail::prefix_hash_chain(prompt);
    const auto points = detail::hash_ladder_points(frontier);
    const auto ladder = detail::prefix_hash_ladder(prompt.token_ids, identity, points);
    expect(ladder.size() == points.size(), "ladder hash count matches point count");
    bool equal = ladder.size() == points.size();
    for (std::size_t j = 0; equal && j < points.size(); ++j) {
        equal = ladder[j] == detail::prefix_hash_at(prompt.token_ids, identity, points[j]) &&
               ladder[j] == chain[points[j]];
    }
    expect(equal, "ladder samples equal per-point hashes and the chain");
    (void)ctx;
    return bad;
}

int test_ladder_diagnostics(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);
    detail::KVRamCache cache(1024 * 1024, host_arena);

    auto leases = materialize(src.state.text_kv, 3);
    const std::vector<std::int32_t> ids{0, 1, 2};
    fill_pages(src.state.text_kv, ids, 3, 0xCD, "ladder fill");
    ctx.synchronize();

    const auto prompt = text_prompt({20, 21, 22, 23, 24, 25});
    const auto retained = retained_prompt(prompt);
    detail::ResidentPrefixIdentity identity;
    const int entry = capture_and_match(
        cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained, 6, ctx.stream,
                                              identity),
                .retained = retained},
        "ladder capture");
    expect(entry != 0, "ladder capture indexed");
    if (entry == 0) { return bad; }

    // A candidate that shares the leading tokens and diverges mid-prompt must stay a miss:
    // the ladder may locate the divergence, but no state-carrying point lies mid-turn, so
    // it must not turn the near match into a hit.
    const auto diverged_prompt = text_prompt({20, 21, 22, 23, 24, 999});
    const auto diverged_retained = retained_prompt(diverged_prompt);
    expect(cache.plan_match(diverged_retained,
                            detail::prefix_hash_chain(diverged_retained)) == std::nullopt,
           "diverged candidate stays a miss (no partial-state reuse)");

    // The identical prompt still hits at the frontier (ladder must not suppress full hits).
    auto match = cache.plan_match(retained, detail::prefix_hash_chain(retained));
    expect(match && match->entry_id == entry && match->reuse_base == 6,
           "identical prompt still hits at the frontier");

    // A prefix-only candidate (frontier beyond its chain) stays a miss as before.
    const auto prefix_prompt = text_prompt({20, 21, 22});
    const auto prefix_retained = retained_prompt(prefix_prompt);
    expect(cache.plan_match(prefix_retained,
                            detail::prefix_hash_chain(prefix_retained)) == std::nullopt,
           "prefix-only candidate stays a miss");
    return bad;
}

int test_fifo_eviction(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);

    auto leases = materialize(src.state.text_kv, 2);
    const std::vector<std::int32_t> ids{0, 1};
    fill_pages(src.state.text_kv, ids, 2, 0xCD, "eviction fill");
    ctx.synchronize();

    const auto prompt_hot = text_prompt({10, 1, 2, 3, 4});
    const auto retained_hot = retained_prompt(prompt_hot);
    const std::uint32_t frontier = 5;
    // Each 2-page record is 1024B for this spec, so the budget fits exactly two records and
    // the third capture forces an eviction of the coldest unpinned record.
    const std::size_t capacity = 2560;
    {
        detail::KVRamCache cache(capacity, host_arena);
        detail::ResidentPrefixIdentity identity;
        // First pass: prove a demonstrated lineage gets a protected record at capture.
        int hot_entry = capture_and_match(
            cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_hot,
                                                  frontier, ctx.stream, identity),
                    .retained = retained_hot},
            "eviction first capture");
        expect(hot_entry != 0, "eviction first capture indexed");
        if (hot_entry == 0) { return bad; }
        cache.claim(hot_entry);
        auto restore_handles = handles(leases);
        detail::RamRestoreTarget restore{
            .text_pages = restore_handles,
            .text_cache = &src.state.text_kv,
            .stream     = ctx.stream,
        };
        cache.unpack_device(hot_entry, restore);
        cache.consume(hot_entry);
        expect(cache.snapshot().entry_count == 0, "exclusive entry consumed");

        // Same lineage again: this record starts out protected.
        int protected_entry = capture_and_match(
            cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_hot,
                                                  frontier, ctx.stream, identity),
                    .retained = retained_hot},
            "eviction protected capture");
        expect(protected_entry != 0, "protected capture indexed");

        const auto prompt_cold1 = text_prompt({10, 20, 30, 40, 50});
        const auto retained_cold1 = retained_prompt(prompt_cold1);
        const int cold1 = capture_and_match(
            cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_cold1,
                                                  frontier, ctx.stream, identity),
                    .retained = retained_cold1},
            "eviction cold1");
        expect(cold1 != 0, "cold1 indexed");

        const auto prompt_cold2 = text_prompt({10, 60, 70, 80, 90});
        const auto retained_cold2 = retained_prompt(prompt_cold2);
        const int cold2 = capture_and_match(
            cache, {.source = CaptureSourceRef(src.state.text_kv, leases, retained_cold2,
                                                  frontier, ctx.stream, identity),
                    .retained = retained_cold2},
            "eviction cold2");
        expect(cold2 != 0, "cold2 indexed");
        expect(cache.snapshot().entry_count == 2, "coldest evicted, two resident");
        expect(cache.snapshot().evictions >= 1, "eviction happened under pressure");

        auto match_cold1 =
            cache.plan_match(retained_cold1, detail::prefix_hash_chain(retained_cold1));
        expect(!match_cold1, "coldest cold record evicted first");
        auto match_hot =
            cache.plan_match(retained_hot, detail::prefix_hash_chain(retained_hot));
        expect(match_hot && match_hot->entry_id == protected_entry,
               "protected record survives pressure");
        auto match_cold2 =
            cache.plan_match(retained_cold2, detail::prefix_hash_chain(retained_cold2));
        expect(match_cold2 && match_cold2->entry_id == cold2, "newest cold record survives");
    }
    return bad;
}

int test_dtor_with_inflight(ninfer::DeviceContext& ctx) {
    int bad = 0;
    KvState src = plan_kv_state(6);
    const std::vector<ninfer::HostKVPageLayout> layouts{
        ninfer::plan_host_kv_page_layout(src.state.text_kv.page_pool().geometry())};
    ninfer::HostKVArena host_arena(4 * 1024 * 1024, layouts);

    auto leases = materialize(src.state.text_kv, 3);
    const std::vector<std::int32_t> ids{0, 1, 2};
    fill_pages(src.state.text_kv, ids, 3, 0xEF, "dtor fill");
    ctx.synchronize();

    const auto prompt   = text_prompt({10, 500, 501});
    const auto retained = retained_prompt(prompt);
    {
        detail::KVRamCache cache(1024 * 1024, host_arena);
        detail::ResidentPrefixIdentity identity;
        detail::RamCaptureSource source =
            make_capture_source(src.state.text_kv, leases, retained, 3, ctx.stream, identity);
        expect(cache.capture(source), "dtor capture accepted");
        // Deliberately no synchronize: the destructor must reap the in-flight copy safely.
    }
    ctx.synchronize();

    // The shared arena is still usable by a later cache.
    {
        detail::KVRamCache cache(1024 * 1024, host_arena);
        detail::ResidentPrefixIdentity identity;
        detail::RamCaptureSource source =
            make_capture_source(src.state.text_kv, leases, retained, 3, ctx.stream, identity);
        expect(cache.capture(source), "arena reused after dtor");
    }
    ctx.synchronize();
    return bad;
}

} // namespace

int main() {
    int count = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || (count_err == cudaSuccess && count == 0)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }

    ninfer::DeviceContext ctx(0);
    int failures = 0;
    failures += test_kv_roundtrip(ctx);
    failures += test_irregular_runs(ctx);
    failures += test_state_image_roundtrip(ctx);
    failures += test_index_match(ctx);
    failures += test_checkpoint_fallback(ctx);
    failures += test_hash_ladder(ctx);
    failures += test_ladder_diagnostics(ctx);
    failures += test_fifo_eviction(ctx);
    failures += test_dtor_with_inflight(ctx);
    if (failures == 0) { std::cout << "OK: kv_ram_cache\n"; }
    return failures == 0 ? 0 : 1;
}
