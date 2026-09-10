#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

namespace {
// Raised inside capture() when the record's host KV page image cannot be allocated although
// the tier budget has room: the shared host KV arena is full of live host-KV offload pages.
// A resource limit, not an invariant error -- the capture is dropped, never surfaced as a
// settle-path exception (pre-r7 this path threw and took the whole settle hook down).
struct RamCaptureDrop {
    const char* reason;
};
} // namespace
namespace {

constexpr std::uint32_t kRamMagic   = 0x4D41524E;
constexpr std::uint32_t kRamVersion = 4;
constexpr std::size_t kSectionCount = 18;
constexpr std::size_t kHostAlign    = 8;
constexpr std::size_t kDeviceAlign  = 256;
constexpr std::size_t kFingerprint  = 56;

std::size_t align_up(std::size_t value, std::size_t align) {
    return (value + align - 1) & ~(align - 1);
}

constexpr std::size_t kLineageAnchorTokens = 256;
constexpr std::size_t kLineageCacheCap     = 2048;

// FNV-1a over the leading tokens of a capture's ledger. This is a lineage hint, not an identity
// check: two different conversations sharing the same system prompt/tool schema prefix will
// collide here, which only makes the cache slightly more generous about what counts as "hot" --
// exact-content correctness for reuse itself is still enforced separately via prefix_matches().
std::uint64_t hash_origin(std::span<const TokenId> ledger) {
    const std::size_t count = std::min(ledger.size(), kLineageAnchorTokens);
    std::uint64_t hash      = 1469598103934665603ULL;
    for (std::size_t i = 0; i < count; ++i) {
        const auto value = static_cast<std::uint32_t>(ledger[i]);
        for (int shift = 0; shift < 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(value >> shift);
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

struct Cursor {
    std::uint8_t* p   = nullptr;
    std::uint8_t* end = nullptr;

    void u8(std::uint8_t v) {
        if (p >= end) { throw std::logic_error("RAM entry write overflow"); }
        *p++ = v;
    }
    void u32(std::uint32_t v) {
        u8(static_cast<std::uint8_t>(v));
        u8(static_cast<std::uint8_t>(v >> 8));
        u8(static_cast<std::uint8_t>(v >> 16));
        u8(static_cast<std::uint8_t>(v >> 24));
    }
    void i32(std::int32_t v) { u32(static_cast<std::uint32_t>(v)); }
    void u64(std::uint64_t v) {
        for (int s = 0; s < 64; s += 8) { u8(static_cast<std::uint8_t>(v >> s)); }
    }
    void i64(std::int64_t v) { u64(static_cast<std::uint64_t>(v)); }
    void bytes(const void* data, std::size_t n) {
        const auto* raw = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { u8(raw[i]); }
    }
};

struct InCursor {
    const std::uint8_t* p   = nullptr;
    const std::uint8_t* end = nullptr;

    [[nodiscard]] std::uint8_t u8() {
        if (p >= end) { throw std::logic_error("RAM entry read overflow"); }
        return *p++;
    }
    [[nodiscard]] std::uint32_t u32() {
        const std::uint32_t a = u8(), b = u8(), c = u8(), d = u8();
        return a | (b << 8) | (c << 16) | (d << 24);
    }
    [[nodiscard]] std::int32_t i32() { return static_cast<std::int32_t>(u32()); }
    [[nodiscard]] std::uint64_t u64() {
        std::uint64_t v = 0;
        for (int s = 0; s < 64; s += 8) { v |= static_cast<std::uint64_t>(u8()) << s; }
        return v;
    }
    [[nodiscard]] std::int64_t i64() { return static_cast<std::int64_t>(u64()); }
    void bytes(void* data, std::size_t n) {
        auto* raw = static_cast<std::uint8_t*>(data);
        for (std::size_t i = 0; i < n; ++i) { raw[i] = u8(); }
    }
    void skip(std::size_t n) {
        if (static_cast<std::size_t>(end - p) < n) {
            throw std::logic_error("RAM entry skip overflow");
        }
        p += n;
    }
};

void write_fingerprint(Cursor& w, const Tensor& plane, PagedKVPlaneOrder order) {
    w.u8(static_cast<std::uint8_t>(plane.dtype));
    w.u8(static_cast<std::uint8_t>(order));
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    w.u8(0);
    for (int i = 0; i < 4; ++i) { w.i32(plane.ne[i]); }
    for (int i = 0; i < 4; ++i) { w.i64(plane.nb[i]); }
}

void check_fingerprint(InCursor& r, const Tensor& plane, PagedKVPlaneOrder order, const char* label) {
    const auto dtype = static_cast<DType>(r.u8());
    const auto stored_order = static_cast<PagedKVPlaneOrder>(r.u8());
    r.skip(6);
    std::int32_t ne[4];
    std::int64_t nb[4];
    for (int i = 0; i < 4; ++i) { ne[i] = r.i32(); }
    for (int i = 0; i < 4; ++i) { nb[i] = r.i64(); }
    if (dtype != plane.dtype || stored_order != order) {
        throw std::logic_error(std::string(label) + " plane dtype/order mismatch");
    }
    for (int i = 0; i < 4; ++i) {
        if (ne[i] != plane.ne[i] || nb[i] != plane.nb[i]) {
            throw std::logic_error(std::string(label) + " plane geometry mismatch");
        }
    }
}

struct HeaderView {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    bool hash_c_valid                     = false;
    std::uint32_t rewrite_frontier        = 0;
    std::uint32_t text_mapped_pages       = 0;
    std::uint32_t backend_mapped_pages    = 0;
    std::uint32_t text_plane_count        = 0;
    std::uint32_t backend_plane_count     = 0;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool has_gdn                          = false;
    bool has_dflash                       = false;
    bool has_state_image                  = false;
    std::uint32_t cyclic_layers           = 0;
    std::uint32_t cyclic_capacity         = 0;
    std::uint32_t cyclic_padded           = 0;
    std::int32_t cyclic_kv_heads          = 0;
    std::int32_t cyclic_head_dim          = 0;
    std::int32_t cyclic_lane_capacity     = 0;
    std::uint64_t tail_hidden_bytes       = 0;
    std::uint64_t gdn_conv_bytes          = 0;
    std::uint64_t gdn_recurrent_bytes     = 0;
    std::uint64_t cyclic_lane_bytes       = 0;
    std::array<std::uint64_t, kSectionCount> offset{};
    std::array<std::uint64_t, kSectionCount> length{};
    std::uint64_t entry_bytes             = 0;
    std::size_t header_bytes              = 0;
};

// Fixed prologue: the scalar fields written by write_fixed_header, then kSectionCount
// offset/length pairs and entry_bytes. Six more sections over version 1.
constexpr std::size_t kFixedHeader = 348 + 6 * 16;

std::size_t header_bytes_for(std::uint32_t text_planes, std::uint32_t backend_planes) {
    return align_up(kFixedHeader + kFingerprint * (text_planes + backend_planes), kHostAlign);
}

HeaderView make_capture_header(const RamCaptureSource& source) {
    if (source.identity == nullptr || source.text_cache == nullptr) {
        throw std::invalid_argument("RAM capture source is incomplete");
    }
    const bool backend_present = !source.backend_pages.empty() || source.backend_cache != nullptr;
    if (backend_present && (source.backend_pages.empty() || source.backend_cache == nullptr)) {
        throw std::invalid_argument("RAM capture backend source is inconsistent");
    }
    if (source.ledger.size() != source.ledger_frontier ||
        source.ledger_frontier != source.execution_frontier + 1) {
        throw std::logic_error("RAM capture ledger frontier is inconsistent");
    }

    HeaderView header;
    header.execution_frontier      = source.execution_frontier;
    header.ledger_frontier         = source.ledger_frontier;
    header.rope_delta              = source.rope_delta;
    header.text_kv_valid           = source.text_kv_valid;
    header.mtp_kv_valid            = source.mtp_kv_valid;
    header.dflash_context_frontier = source.dflash_context_frontier;
    header.tail_hidden_valid       = source.tail_hidden_valid;
    header.rewrite_valid           = source.rewrite_valid;
    header.rewrite_kind            = source.rewrite_kind;
    header.hash_c_valid            = source.hash_c_valid;
    header.rewrite_frontier        = source.rewrite_frontier;
    header.text_mapped_pages     = static_cast<std::uint32_t>(source.text_pages.size());
    header.backend_mapped_pages  = static_cast<std::uint32_t>(source.backend_pages.size());
    header.text_plane_count      =
        static_cast<std::uint32_t>(source.text_cache->page_pool().plane_count());
    header.backend_plane_count =
        source.backend_cache ? static_cast<std::uint32_t>(source.backend_cache->page_pool().plane_count()) : 0;
    header.hash_f                  = source.hash_f;
    header.hash_c                  = source.hash_c;
    header.has_state_image         = source.state_image != nullptr;
    if (source.state_image != nullptr) {
        // State-image record: the state half is complete StateImage payloads, so the
        // per-region scalars stay zero; the geometry is the layout itself.
        const StateImageHostLayout& lay = source.state_image->host_layout();
        header.has_gdn               = true; // every state image carries linear (GDN) state
        header.has_dflash            = lay.dflash_local_k.has_value();
        if (const auto& dflash = lay.spec.dflash_local) {
            header.cyclic_layers     = dflash->layers;
            header.cyclic_capacity   = dflash->capacity;
            header.cyclic_kv_heads   = dflash->kv_heads;
            header.cyclic_head_dim   = dflash->head_dim;
        }
    } else {
        // No state image: the GDN and DFlash local state live in the baseline's unified state
        // image, so a record without one carries no linear-attention or cyclic state; only
        // the standalone hidden tensors (sections 8/9) are carried.
        if (source.tail_hidden != nullptr) { header.tail_hidden_bytes = source.tail_hidden->bytes(); }
    }
    return header;
}

std::array<std::size_t, kSectionCount> finalize_capture_layout(const RamCaptureSource& source,
                                                                HeaderView& header) {
    std::array<std::size_t, kSectionCount> lengths{};
    std::array<std::size_t, kSectionCount> aligns{};
    lengths[0] = source.ledger.size() * sizeof(TokenId);
    lengths[1] = source.identity->packed_bytes();
    // The paged image lives in the host KV arena (capture() allocates one arena allocation per
    // non-empty page span and stream-copies into it); the flat block carries no image bytes.
    lengths[2] = 0;
    lengths[3] = 0;
    if (source.state_image != nullptr) {
        // The state half is complete StateImage payloads: sections 4/5 each hold one
        // image_bytes-sized image laid out per StateImageHostLayout; sections 6..11 unused.
        const std::size_t image_bytes = source.state_image->host_layout().image_bytes;
        lengths[4] = image_bytes;
        lengths[5] = source.rewrite_valid ? image_bytes : 0;
    } else {
        // GDN (sections 4..7) and DFlash (10/11) are state-image-only; without a state image
        // only the standalone hidden tensors (8/9) are carried.
        lengths[8] = source.tail_hidden ? source.tail_hidden->bytes() : 0;
        lengths[9] = source.rewrite_valid && source.rewrite_checkpoint_hidden
                         ? source.rewrite_checkpoint_hidden->bytes()
                         : 0;
    }
    aligns[0] = kHostAlign;
    aligns[1] = kHostAlign;
    for (std::size_t i = 2; i < kSectionCount; ++i) { aligns[i] = kDeviceAlign; }

    header.header_bytes = header_bytes_for(header.text_plane_count, header.backend_plane_count);
    std::size_t cursor  = header.header_bytes;
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        if (lengths[i] == 0) { continue; }
        cursor           = align_up(cursor, aligns[i]);
        header.offset[i] = cursor;
        header.length[i] = lengths[i];
        cursor += lengths[i];
    }
    header.entry_bytes = align_up(cursor, kDeviceAlign);
    return lengths;
}

void write_fixed_header(Cursor& w, const HeaderView& h) {
    w.u32(kRamMagic);
    w.u32(kRamVersion);
    w.u32(h.execution_frontier);
    w.u32(h.ledger_frontier);
    w.i32(h.rope_delta);
    w.u32(h.text_kv_valid);
    w.u32(h.mtp_kv_valid);
    w.u32(h.dflash_context_frontier);
    w.u8(h.tail_hidden_valid ? 1 : 0);
    w.u8(h.rewrite_valid ? 1 : 0);
    w.u8(static_cast<std::uint8_t>(h.rewrite_kind));
    w.u8(h.hash_c_valid ? 1 : 0);
    w.u32(h.rewrite_frontier);
    w.u32(h.text_mapped_pages);
    w.u32(h.backend_mapped_pages);
    w.u32(h.text_plane_count);
    w.u32(h.backend_plane_count);
    w.u64(h.hash_f.lo);
    w.u64(h.hash_f.hi);
    w.u64(h.hash_c.lo);
    w.u64(h.hash_c.hi);
    w.u8(h.has_gdn ? 1 : 0);
    w.u8(h.has_dflash ? 1 : 0);
    w.u8(h.has_state_image ? 1 : 0);
    w.u8(0);
    w.u32(h.cyclic_layers);
    w.u32(h.cyclic_capacity);
    w.u32(h.cyclic_padded);
    w.i32(h.cyclic_kv_heads);
    w.i32(h.cyclic_head_dim);
    w.i32(h.cyclic_lane_capacity);
    w.u64(h.tail_hidden_bytes);
    w.u64(h.gdn_conv_bytes);
    w.u64(h.gdn_recurrent_bytes);
    w.u64(h.cyclic_lane_bytes);
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        w.u64(h.offset[i]);
        w.u64(h.length[i]);
    }
    w.u64(h.entry_bytes);
}

HeaderView read_header(const void* block, std::size_t bytes) {
    if (block == nullptr || bytes < kFixedHeader) {
        throw std::logic_error("RAM entry header is truncated");
    }
    const auto* raw = static_cast<const std::uint8_t*>(block);
    InCursor r{raw, raw + bytes};
    HeaderView h;
    if (r.u32() != kRamMagic || r.u32() != kRamVersion) {
        throw std::logic_error("RAM entry magic/version mismatch");
    }
    h.execution_frontier      = r.u32();
    h.ledger_frontier         = r.u32();
    h.rope_delta              = r.i32();
    h.text_kv_valid           = r.u32();
    h.mtp_kv_valid            = r.u32();
    h.dflash_context_frontier = r.u32();
    h.tail_hidden_valid       = r.u8() != 0;
    h.rewrite_valid           = r.u8() != 0;
    h.rewrite_kind            = static_cast<RewriteCheckpointKind>(r.u8());
    h.hash_c_valid            = r.u8() != 0;
    h.rewrite_frontier        = r.u32();
    h.text_mapped_pages       = r.u32();
    h.backend_mapped_pages    = r.u32();
    h.text_plane_count        = r.u32();
    h.backend_plane_count     = r.u32();
    h.hash_f.lo               = r.u64();
    h.hash_f.hi               = r.u64();
    h.hash_c.lo               = r.u64();
    h.hash_c.hi               = r.u64();
    h.has_gdn                 = r.u8() != 0;
    h.has_dflash              = r.u8() != 0;
    h.has_state_image = r.u8() != 0;
    r.u8();
    h.cyclic_layers           = r.u32();
    h.cyclic_capacity         = r.u32();
    h.cyclic_padded           = r.u32();
    h.cyclic_kv_heads         = r.i32();
    h.cyclic_head_dim         = r.i32();
    h.cyclic_lane_capacity    = r.i32();
    h.tail_hidden_bytes       = r.u64();
    h.gdn_conv_bytes          = r.u64();
    h.gdn_recurrent_bytes     = r.u64();
    h.cyclic_lane_bytes       = r.u64();
    for (std::size_t i = 0; i < kSectionCount; ++i) {
        h.offset[i] = r.u64();
        h.length[i] = r.u64();
    }
    h.entry_bytes  = r.u64();
    h.header_bytes = header_bytes_for(h.text_plane_count, h.backend_plane_count);
    if (h.entry_bytes > bytes || h.header_bytes > bytes) {
        throw std::logic_error("RAM entry header size is inconsistent");
    }
    return h;
}

const std::uint8_t* section_ptr(const void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<const std::uint8_t*>(block) + header.offset[index];
}

std::uint8_t* section_ptr(void* block, const HeaderView& header, std::size_t index) {
    if (header.length[index] == 0) { return nullptr; }
    return static_cast<std::uint8_t*>(block) + header.offset[index];
}

void verify_pool(InCursor& r, const DeviceKVPagePool& pool, std::uint32_t stored_planes,
                 const char* label) {
    if (stored_planes != pool.plane_count()) {
        throw std::logic_error(std::string(label) + " plane count mismatch");
    }
    for (std::uint32_t i = 0; i < stored_planes; ++i) {
        check_fingerprint(r, pool.plane(i), pool.geometry().device_plane_order, label);
    }
}


RamRestoredHost host_from_header(const void* block, const HeaderView& header) {
    RamRestoredHost out;
    out.execution_frontier      = header.execution_frontier;
    out.ledger_frontier         = header.ledger_frontier;
    out.rope_delta              = header.rope_delta;
    out.text_kv_valid           = header.text_kv_valid;
    out.mtp_kv_valid            = header.mtp_kv_valid;
    out.dflash_context_frontier = header.dflash_context_frontier;
    out.tail_hidden_valid       = header.tail_hidden_valid;
    out.rewrite_valid           = header.rewrite_valid;
    out.rewrite_kind            = header.rewrite_kind;
    out.rewrite_frontier        = header.rewrite_frontier;
    out.backend_image_present   = header.backend_mapped_pages > 0;
    const auto* ledger = section_ptr(block, header, 0);
    if (header.length[0] != header.ledger_frontier * sizeof(TokenId)) {
        throw std::logic_error("RAM entry ledger size mismatch");
    }
    out.ledger.resize(header.ledger_frontier);
    if (!out.ledger.empty()) {
        std::memcpy(out.ledger.data(), ledger, header.length[0]);
    }
    const auto* identity = section_ptr(block, header, 1);
    out.identity.unpack(identity, static_cast<std::size_t>(header.length[1]));
    return out;
}

} // namespace

KVRamCache::KVRamCache(std::size_t capacity_bytes, HostKVArena& host_kv_arena)
    : arena_(capacity_bytes), host_kv_arena_(&host_kv_arena) {}

KVRamCache::~KVRamCache() {
    reap_retired(true);
    std::vector<std::uint64_t> ids(fifo_.begin(), fifo_.end());
    for (std::uint64_t id : ids) {
        auto it = records_.find(id);
        if (it == records_.end()) { continue; }
        if (it->second.copies_start != nullptr) {
            (void)cudaEventDestroy(it->second.copies_start);
            it->second.copies_start = nullptr;
        }
        if (it->second.copies_done != nullptr) {
            (void)cudaEventSynchronize(it->second.copies_done);
            (void)cudaEventDestroy(it->second.copies_done);
            it->second.copies_done = nullptr;
        }
    }
    records_.clear();
    fifo_.clear();
}

KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) {
    const auto it = records_.find(entry_id);
    if (it == records_.end()) { throw std::logic_error("RAM cache entry id is unknown"); }
    return it->second;
}

const KVRamCache::Record& KVRamCache::require(std::uint64_t entry_id) const {
    const auto it = records_.find(entry_id);
    if (it == records_.end()) { throw std::logic_error("RAM cache entry id is unknown"); }
    return it->second;
}

void KVRamCache::destroy_record(std::uint64_t entry_id, bool count_eviction) {
    auto it = records_.find(entry_id);
    if (it == records_.end()) { return; }
    orphaned_save_seconds_ += harvest_record(it->second);
    wait_copies(it->second);
    if (it->second.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(it->second.copies_start));
        it->second.copies_start = nullptr;
    }
    if (it->second.copies_done != nullptr) {
        CUDA_CHECK(cudaEventDestroy(it->second.copies_done));
        it->second.copies_done = nullptr;
    }
    it->second.copies_timed = false;
    if (it->second.block != nullptr) {
        arena_.free(it->second.block);
        // A retired record is erased from records_ at retire time (where the footprint was
        // already discharged); only a live record reaches here with its block intact.
        host_footprint_bytes_ -= it->second.bytes + it->second.host_kv_bytes;
        host_kv_bytes_used_   -= it->second.host_kv_bytes;
        if (count_eviction) {
            std::fprintf(stderr, "[t8] evict id=%llu block=%zu kv=%zu\n",
                         static_cast<unsigned long long>(entry_id), it->second.bytes,
                         it->second.host_kv_bytes);
            std::fflush(stderr);
        }
    }
    records_.erase(it);
    fifo_.erase(std::remove(fifo_.begin(), fifo_.end(), entry_id), fifo_.end());
    if (count_eviction) { ++evictions_; }
    bump_version();
}

void KVRamCache::begin_copies(Record& record, cudaStream_t stream) {
    if (record.copies_start != nullptr) {
        CUDA_CHECK(cudaEventDestroy(record.copies_start));
        record.copies_start = nullptr;
    }
    CUDA_CHECK(cudaEventCreate(&record.copies_start));
    CUDA_CHECK(cudaEventRecord(record.copies_start, stream));
    record.copies_timed = false;
}

void KVRamCache::record_copies(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) { CUDA_CHECK(cudaEventCreate(&record.copies_done)); }
    CUDA_CHECK(cudaEventRecord(record.copies_done, stream));
    record.copies_timed = record.copies_start != nullptr;
}

double KVRamCache::harvest_record(Record& record) {
    if (!record.copies_timed || record.copies_start == nullptr || record.copies_done == nullptr) {
        return 0;
    }
    wait_copies(record);
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, record.copies_start, record.copies_done));
    record.copies_timed = false;
    CUDA_CHECK(cudaEventDestroy(record.copies_start));
    record.copies_start = nullptr;
    return static_cast<double>(milliseconds) / 1000.0;
}

KvRamCopySeconds KVRamCache::harvest_copy_seconds() {
    KvRamCopySeconds out;
    out.save += orphaned_save_seconds_;
    save_seconds_ += orphaned_save_seconds_;
    orphaned_save_seconds_ = 0;
    for (std::uint64_t id : pending_save_ids_) {
        const auto it = records_.find(id);
        if (it == records_.end()) { continue; }
        const double seconds = harvest_record(it->second);
        out.save += seconds;
        save_seconds_ += seconds;
    }
    pending_save_ids_.clear();
    if (pending_load_id_) {
        const auto it = records_.find(*pending_load_id_);
        if (it != records_.end()) {
            const double seconds = harvest_record(it->second);
            out.load += seconds;
            load_seconds_ += seconds;
        }
        pending_load_id_.reset();
    }
    return out;
}

void KVRamCache::wait_copies(Record& record) {
    if (record.copies_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(record.copies_done)); }
}

void KVRamCache::wait_copies_on_stream(Record& record, cudaStream_t stream) {
    if (record.copies_done == nullptr) { return; }
    if (stream != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, record.copies_done, 0));
        return;
    }
    wait_copies(record);
}

void KVRamCache::retire_record(Record& record) {
    RetiredCopy item;
    item.block             = record.block;
    item.copies_done       = record.copies_done;
    item.text_host_kv      = std::move(record.text_host_kv);
    item.backend_host_kv   = std::move(record.backend_host_kv);
    record.block           = nullptr;
    record.copies_done     = nullptr;
    // The record leaves records_ the moment ownership moves to retired_; discharge the
    // footprint now (the underlying memory is freed later by reap_retired).
    host_footprint_bytes_ -= record.bytes + record.host_kv_bytes;
    host_kv_bytes_used_   -= record.host_kv_bytes;
    retired_.push_back(std::move(item));
}

void KVRamCache::reap_retired(bool block) {
    std::size_t keep = 0;
    for (RetiredCopy& item : retired_) {
        if (item.copies_done != nullptr) {
            if (!block) {
                const cudaError_t ready = cudaEventQuery(item.copies_done);
                if (ready == cudaErrorNotReady) {
                    retired_[keep++] = std::move(item);
                    continue;
                }
                CUDA_CHECK(ready);
            } else {
                CUDA_CHECK(cudaEventSynchronize(item.copies_done));
            }
            CUDA_CHECK(cudaEventDestroy(item.copies_done));
            item.copies_done = nullptr;
        }
        if (item.block != nullptr) {
            arena_.free(item.block);
            item.block = nullptr;
        }
    }
    retired_.resize(keep);
}

bool KVRamCache::lineage_is_hot(std::uint64_t origin_hash) const {
    const auto it = lineage_hits_.find(origin_hash);
    return it != lineage_hits_.end() && it->second > 0;
}

void KVRamCache::note_lineage_hit(std::uint64_t origin_hash) {
    const auto [it, inserted] = lineage_hits_.try_emplace(origin_hash, 0);
    if (inserted) {
        lineage_order_.push_back(origin_hash);
        while (lineage_order_.size() > kLineageCacheCap) {
            lineage_hits_.erase(lineage_order_.front());
            lineage_order_.pop_front();
        }
    }
    if (it->second < std::numeric_limits<std::uint32_t>::max()) { ++it->second; }
}

namespace {
constexpr std::uint32_t kMaxDynamicBoundaryRecords = 2;
} // namespace

void KVRamCache::touch(std::uint64_t entry_id) {
    const auto it = std::find(fifo_.begin(), fifo_.end(), entry_id);
    if (it == fifo_.end()) { return; }
    fifo_.erase(it);
    fifo_.push_back(entry_id);
}

bool KVRamCache::make_room_for_dynamic_boundary() {
    std::uint32_t live = 0;
    for (std::uint64_t id : fifo_) {
        const auto it = records_.find(id);
        if (it != records_.end() && it->second.capture_kind == RamCaptureKind::DynamicBoundary) {
            ++live;
        }
    }
    if (live < kMaxDynamicBoundaryRecords) { return true; }
    // fifo_ is touched on every hit (see touch()), so the front-most match here is the coldest
    // unclaimed DynamicBoundary record, not merely the oldest by capture time.
    for (std::uint64_t id : fifo_) {
        const auto it = records_.find(id);
        if (it != records_.end() && it->second.capture_kind == RamCaptureKind::DynamicBoundary &&
            it->second.claims == 0) {
            destroy_record(id, true);
            return true;
        }
    }
    return false;
}

void KVRamCache::evict_unpinned() {
    // Class is durable across a VRAM spill. Classifier records and unproven non-main records
    // churn first, demonstrated ordinary agent state next, and main conversation state last.
    // Claims remain absolute pins; the final main pass preserves capture forward progress.
    for (int rank = 0; rank != 3; ++rank) {
        for (std::uint64_t id : fifo_) {
            const auto it = records_.find(id);
            if (it == records_.end() || it->second.claims != 0) { continue; }
            const Record& record = it->second;
            const int record_rank = record.owner_class == RequestClass::Main
                                        ? 2
                                        : (record.owner_class == RequestClass::Classifier ||
                                                   !record.protected_tier
                                               ? 0
                                               : 1);
            if (record_rank == rank) {
                destroy_record(id, true);
                return;
            }
        }
    }
}

void KVRamCache::claim(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims != 0 && !record.multi_claim) {
        throw std::logic_error("RAM cache entry is already claimed");
    }
    ++record.claims;
    bump_version();
}

void KVRamCache::release(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims == 0) { throw std::logic_error("RAM cache entry is not claimed"); }
    --record.claims;
    bump_version();
}

void KVRamCache::consume(std::uint64_t entry_id) {
    Record& record = require(entry_id);
    if (record.claims == 0) { throw std::logic_error("RAM cache consume requires a claimed entry"); }
    ++restores_;
    note_lineage_hit(record.origin_hash);
    touch(entry_id);
    --record.claims;
    if (record.multi_claim) {
        // Another sibling may still want to restore from this same entry -- drop this claim but
        // keep the record, which stays matchable throughout. It ages out through ordinary
        // FIFO/tiered eviction like any other record once nothing claims it further.
        record.protected_tier = true;
        bump_version();
        return;
    }
    retire_record(record);
    records_.erase(entry_id);
    fifo_.erase(std::remove(fifo_.begin(), fifo_.end(), entry_id), fifo_.end());
    bump_version();
    reap_retired(false);
}

std::optional<RamMatch> KVRamCache::plan_match(const PreparedPromptData& prompt,
                                               std::span<const PrefixHash128> hash_chain) {
    std::optional<RamMatch> best;
    // No-reuse diagnostics: the record whose ladder agrees deepest with the candidate, so a
    // miss can be journalled with the divergence point instead of a bare "nothing matched".
    std::uint64_t near_id    = 0;
    std::uint32_t near_point = 0;
    for (std::uint64_t id : fifo_) {
        const Record& record = require(id);
        // A claimed exclusive record is spoken for and vanishes on its claimant's restore. A
        // multi_claim record stays matchable while claimed: several siblings of one burst
        // legitimately restore from the same snapshot, and hiding it from the ones that arrive
        // while a first claim is outstanding is what used to make each of them capture its own
        // duplicate of the same lane.
        if (record.claims != 0 && !record.multi_claim) { continue; }
        RamMatch candidate;
        candidate.entry_id = id;
        // Ladder: matching points form a run from index 0 (a hash equal at point k implies
        // agreement over [0:k] of this record's ledger), so scan ascending until the first
        // disagreement. The point is diagnostic only -- the reusable gates below are
        // unchanged (a frontier/checkpoint hash must additionally exact-match with state).
        std::uint32_t ladder_point = 0;
        for (std::size_t j = 0; j < record.ladder_points.size(); ++j) {
            const std::uint32_t p = record.ladder_points[j];
            if (p >= hash_chain.size() || hash_chain[p] != record.ladder_hashes[j]) { break; }
            ladder_point = p;
        }
        if (near_id == 0 || ladder_point > near_point) {
            near_id    = id;
            near_point = ladder_point;
        }
        const bool frontier_hash =
            record.execution_frontier > 0 && record.execution_frontier < hash_chain.size() &&
            hash_chain[record.execution_frontier] == record.hash_f;
        const bool checkpoint_hash =
            record.hash_c_valid && record.checkpoint_frontier > 0 &&
            record.checkpoint_frontier < hash_chain.size() &&
            hash_chain[record.checkpoint_frontier] == record.hash_c;
        if (!frontier_hash && !checkpoint_hash) { continue; }

        const HeaderView header = read_header(record.block, record.bytes);
        const RamRestoredHost host = host_from_header(record.block, header);
        if (frontier_hash) {
            ++exact_comparisons_;
            if (prefix_matches(prompt, host.ledger, host.identity, record.execution_frontier)) {
                candidate.reuse      = PrefixReusePath::AppendAtFrontier;
                candidate.reuse_base = record.execution_frontier;
            }
        }
        if (candidate.reuse_base == 0 && checkpoint_hash) {
            ++exact_comparisons_;
            if (prefix_matches(prompt, host.ledger, host.identity, record.checkpoint_frontier)) {
                candidate.reuse      = record.checkpoint_path;
                candidate.reuse_base = record.checkpoint_frontier;
            }
        }
        if (candidate.reuse_base == 0) { continue; }
        if (!best || candidate.reuse_base > best->reuse_base) { best = candidate; }
    }
    if (best.has_value()) {
        std::fprintf(stderr, "[t8] match hit id=%llu base=%u entries=%zu\n",
                     static_cast<unsigned long long>(best->entry_id), best->reuse_base,
                     fifo_.size());
        std::fflush(stderr);
    } else if (!fifo_.empty()) {
        int first_div       = -1;
        std::uint64_t exp_token = 0, act_token = 0;
        if (near_id != 0 && near_point > 0) {
            // Bounded scan from the deepest agreed ladder point: where exactly did this
            // candidate stop matching the near record, and what token did it swap in?
            // exp = the record's (expected) token, act = the candidate's (actual) token.
            const RamRestoredHost host = load_host(near_id);
            auto diverged = first_divergence(
                prompt, std::span<const TokenId>(host.ledger), host.identity, near_point,
                static_cast<std::size_t>(near_point) + 512);
            if (diverged) {
                first_div = static_cast<int>(*diverged);
                exp_token = host.ledger[*diverged];
                act_token = prompt.token_ids[*diverged];
            }
        }
        std::fprintf(stderr,
                     "[t8] match miss entries=%zu near=%u id=%llu first_div=%d exp=%llu act=%llu\n",
                     fifo_.size(), near_point, static_cast<unsigned long long>(near_id),
                     first_div, static_cast<unsigned long long>(exp_token),
                     static_cast<unsigned long long>(act_token));
        std::fflush(stderr);
    }
    return best;
}

RamRestoredHost KVRamCache::load_host(std::uint64_t entry_id) const {
    const Record& record = require(entry_id);
    return host_from_header(record.block, read_header(record.block, record.bytes));
}

KvRamSnapshot KVRamCache::snapshot() const noexcept {
    std::size_t used = 0;
    for (const auto& entry : records_) { used += entry.second.bytes; }
    return KvRamSnapshot{
        .capacity_bytes = arena_.capacity(),
        .used_bytes     = used,
        .kv_image_bytes = host_kv_bytes_used_,
        .entry_count    = records_.size(),
        .captures       = captures_,
        .restores       = restores_,
        .evictions      = evictions_,
        .drops          = drops_,
        .save_seconds   = save_seconds_,
        .load_seconds   = load_seconds_,
    };
}

std::size_t KVRamCache::capture_bytes(const RamCaptureSource& source) const {
    HeaderView header = make_capture_header(source);
    (void)finalize_capture_layout(source, header);
    return header.entry_bytes;
}

bool KVRamCache::has_room_for_dynamic_boundary_without_eviction() const noexcept {
    std::uint32_t live = 0;
    for (std::uint64_t id : fifo_) {
        const auto it = records_.find(id);
        if (it != records_.end() && it->second.capture_kind == RamCaptureKind::DynamicBoundary) {
            ++live;
        }
    }
    return live < kMaxDynamicBoundaryRecords;
}

bool KVRamCache::can_capture_without_eviction(const RamCaptureSource& source) {
    reap_retired(false);
    const std::size_t bytes = capture_bytes(source);
    if (!arena_.can_alloc(bytes, kDeviceAlign)) { return false; }
    return source.capture_kind != RamCaptureKind::DynamicBoundary ||
           has_room_for_dynamic_boundary_without_eviction();
}

bool KVRamCache::capture(const RamCaptureSource& source, RamCapturePolicy policy) {
    if (source.identity == nullptr || source.text_cache == nullptr) {
        throw std::invalid_argument("RAM capture source is incomplete");
    }
    const bool backend_present = !source.backend_pages.empty() || source.backend_cache != nullptr;
    if (backend_present && (source.backend_pages.empty() || source.backend_cache == nullptr)) {
        throw std::invalid_argument("RAM capture backend source is inconsistent");
    }
    if (source.ledger.size() != source.ledger_frontier ||
        source.ledger_frontier != source.execution_frontier + 1) {
        throw std::logic_error("RAM capture ledger frontier is inconsistent");
    }

    HeaderView header = make_capture_header(source);
    const std::array<std::size_t, kSectionCount> lengths =
        finalize_capture_layout(source, header);
    const std::size_t header_bytes = header.header_bytes;

    // The record's paged-KV image half lands in the lane's shared host KV arena. Compute its
    // size up front so the tier budget (capacity_bytes) bounds the tier's TOTAL host
    // commitment -- flat block plus page images -- rather than the flat block alone.
    // Pre-r7 the page images were unaccounted, which over-committed host RAM on small boxes
    // (the r6 window: records pushed the host into zram thrash and collapsed decode).
    const HostKVPageLayout* text_layout = nullptr;
    const HostKVPageLayout* backend_layout = nullptr;
    std::size_t text_kv_bytes = 0;
    std::size_t backend_kv_bytes = 0;
    if (!source.text_pages.empty()) {
        text_layout = host_kv_arena_->layout_for(source.text_cache->page_pool().geometry());
        if (text_layout == nullptr) {
            throw std::logic_error("RAM capture text pool has no arena layout");
        }
        text_kv_bytes = text_layout->page_stride * source.text_pages.size();
    }
    if (!source.backend_pages.empty()) {
        backend_layout = host_kv_arena_->layout_for(source.backend_cache->page_pool().geometry());
        if (backend_layout == nullptr) {
            throw std::logic_error("RAM capture backend pool has no arena layout");
        }
        backend_kv_bytes = backend_layout->page_stride * source.backend_pages.size();
    }

    const std::size_t prospective_bytes =
        header.entry_bytes + text_kv_bytes + backend_kv_bytes;
    if (host_footprint_bytes_ + prospective_bytes > arena_.capacity()) {
        // The record does not fit the tier budget: reclaim coldest unpinned records (which also
        // frees their host KV arena images, giving arena room back to live host-KV offload)
        // until it does, then drop if the budget cannot be met at all.
        for (;;) {
            const std::size_t before = records_.size();
            evict_unpinned();
            if (records_.size() == before) { break; }
            if (host_footprint_bytes_ + prospective_bytes <= arena_.capacity()) { break; }
        }
        if (host_footprint_bytes_ + prospective_bytes > arena_.capacity()) {
            ++drops_;
            bump_version();
            std::fprintf(stderr,
                         "[t8] capture drop reason=budget entries=%zu host_used=%zu "
                         "prospective=%zu cap=%zu\n",
                         records_.size(), host_footprint_bytes_,
                         host_footprint_bytes_ + prospective_bytes, arena_.capacity());
            std::fflush(stderr);
            return false;
        }
    }
    // Enforced only once the entry is known to fit the arena at all -- evicting a live
    // DynamicBoundary record for a capture that could never succeed regardless (oversized entry,
    // or a source invariant that would have thrown above) would waste it for nothing.
    if (source.capture_kind == RamCaptureKind::DynamicBoundary &&
        (policy == RamCapturePolicy::PreserveExisting
             ? !has_room_for_dynamic_boundary_without_eviction()
             : !make_room_for_dynamic_boundary())) {
        ++drops_;
        bump_version();
        return false;
    }

    reap_retired(false);
    void* block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    if (block == nullptr) {
        reap_retired(true);
        block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    }
    while (block == nullptr && policy == RamCapturePolicy::AllowEviction) {
        const std::size_t before = records_.size();
        evict_unpinned();
        if (records_.size() == before) {
            ++drops_;
            bump_version();
            return false;
        }
        block = arena_.try_alloc(header.entry_bytes, kDeviceAlign);
    }
    if (block == nullptr) {
        ++drops_;
        bump_version();
        return false;
    }

    bool copies_launched = false;
    std::uint64_t live_id  = 0;
    cudaEvent_t copies_start = nullptr;
    HostKVAllocation text_host_kv;
    HostKVAllocation backend_host_kv;
    try {
        auto* raw = static_cast<std::uint8_t*>(block);
        std::memset(raw, 0, header_bytes);
        Cursor w{raw, raw + header_bytes};
        write_fixed_header(w, header);
        const DeviceKVPagePool& text_pool = source.text_cache->page_pool();
        for (std::uint32_t i = 0; i < header.text_plane_count; ++i) {
            write_fingerprint(w, text_pool.plane(i), text_pool.geometry().device_plane_order);
        }
        if (source.backend_cache != nullptr) {
            const DeviceKVPagePool& backend_pool = source.backend_cache->page_pool();
            for (std::uint32_t i = 0; i < header.backend_plane_count; ++i) {
                write_fingerprint(w, backend_pool.plane(i),
                                  backend_pool.geometry().device_plane_order);
            }
        }

        if (lengths[0] != 0) {
            std::memcpy(raw + header.offset[0], source.ledger.data(), lengths[0]);
        }
        if (lengths[1] != 0) { source.identity->pack(raw + header.offset[1]); }
        const auto start_device_copies = [&] {
            if (copies_start != nullptr) { return; }
            CUDA_CHECK(cudaEventCreate(&copies_start));
            CUDA_CHECK(cudaEventRecord(copies_start, source.stream));
        };
        // The paged image is captured straight into a host KV arena allocation (one per
        // non-empty page span) via the device pool's stream-async host copy -- the same
        // demotion-copy path the extent store uses; the flat block carries no image bytes.
        if (!source.text_pages.empty()) {
            auto alloc = host_kv_arena_->allocate(
                *text_layout, static_cast<std::uint32_t>(source.text_pages.size()));
            if (!alloc) {
                throw RamCaptureDrop{"host-kv-arena-full"};
            }
            text_host_kv = std::move(*alloc);
            auto text_view = host_kv_arena_->writable_view(text_host_kv);
            start_device_copies();
            source.text_cache->page_pool().copy_to_host(source.text_pages, text_view, source.stream);
            copies_launched = true;
        }
        if (!source.backend_pages.empty()) {
            auto alloc = host_kv_arena_->allocate(
                *backend_layout, static_cast<std::uint32_t>(source.backend_pages.size()));
            if (!alloc) {
                throw RamCaptureDrop{"host-kv-arena-full"};
            }
            backend_host_kv = std::move(*alloc);
            auto backend_view = host_kv_arena_->writable_view(backend_host_kv);
            start_device_copies();
            source.backend_cache->page_pool().copy_to_host(source.backend_pages, backend_view,
                                                           source.stream);
            copies_launched = true;
        }
        if (source.state_image != nullptr) {
            // One async device->host copy per slot lands the complete StateImage directly in
            // the capture block (sections 4/5) -- the same stream-async pattern as the paged KV
            // packs; no per-type pack, no pinned-pool round trip. The legacy per-type blocks
            // below are no-ops here (their source pointers are null, sections 6..11 zero).
            start_device_copies();
            HostStateImageView current_view{reinterpret_cast<std::byte*>(raw + header.offset[4]),
                                            &source.state_image->host_layout()};
            source.state_image->copy_to_host(source.state_slot, current_view, source.stream);
            copies_launched = true;
            if (lengths[5] != 0) {
                HostStateImageView checkpoint_view{reinterpret_cast<std::byte*>(raw + header.offset[5]),
                                                    &source.state_image->host_layout()};
                source.state_image->copy_to_host(source.state_checkpoint_slot, checkpoint_view,
                                                 source.stream);
                copies_launched = true;
            }
        }
        if (source.tail_hidden != nullptr && lengths[8] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[8], source.tail_hidden->data, lengths[8],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }
        if (source.rewrite_checkpoint_hidden != nullptr && lengths[9] != 0) {
            start_device_copies();
            CUDA_CHECK(cudaMemcpyAsync(raw + header.offset[9],
                                       source.rewrite_checkpoint_hidden->data, lengths[9],
                                       cudaMemcpyDeviceToHost, source.stream));
            copies_launched = true;
        }

        if (next_id_ == 0) { throw std::logic_error("RAM cache entry id overflow"); }
        Record record;
        record.id                  = next_id_++;
        record.hash_f              = source.hash_f;
        record.hash_c              = source.hash_c;
        record.hash_c_valid        = source.hash_c_valid;
        record.execution_frontier  = source.execution_frontier;
        record.checkpoint_frontier = source.rewrite_frontier;
        record.checkpoint_valid    = source.rewrite_valid;
        record.checkpoint_path     = source.rewrite_kind == RewriteCheckpointKind::TurnClosure
                                         ? PrefixReusePath::RestoreTurnCheckpoint
                                         : PrefixReusePath::RestoreResponseCheckpoint;
        record.block               = block;
        record.bytes               = header.entry_bytes;
        record.host_kv_bytes       = text_kv_bytes + backend_kv_bytes;
        record.text_host_kv    = std::move(text_host_kv);
        record.backend_host_kv = std::move(backend_host_kv);
        record.origin_hash         = hash_origin(source.ledger);
        record.protected_tier      = lineage_is_hot(record.origin_hash);
        record.multi_claim         = source.multi_claim;
        record.owner_class         = source.owner_class;
        record.capture_kind        = source.capture_kind;
        if (source.identity != nullptr) {
            record.ladder_points = hash_ladder_points(source.execution_frontier);
            record.ladder_hashes =
                prefix_hash_ladder(source.ledger, *source.identity, record.ladder_points);
        }
        record.copies_start        = copies_start;
        copies_start               = nullptr;
        const auto [it, inserted]  = records_.emplace(record.id, std::move(record));
        if (!inserted) { throw std::logic_error("RAM cache entry id already exists"); }
        live_id = it->second.id;
        host_footprint_bytes_ += it->second.bytes + it->second.host_kv_bytes;
        host_kv_bytes_used_   += it->second.host_kv_bytes;
        fifo_.push_back(it->second.id);
        record_copies(it->second, source.stream);
        pending_save_ids_.push_back(it->second.id);
        ++captures_;
        bump_version();
        std::fprintf(stderr,
                     "[t8] capture ok id=%llu frontier=%u kind=%d block=%zu kv=%zu "
                     "host_used=%zu entries=%zu\n",
                     static_cast<unsigned long long>(live_id), it->second.execution_frontier,
                     static_cast<int>(it->second.capture_kind), it->second.bytes,
                     it->second.host_kv_bytes, host_footprint_bytes_, records_.size());
        std::fflush(stderr);
        return true;
    } catch (const RamCaptureDrop& drop) {
        if (copies_launched) {
            if (source.stream != nullptr) {
                (void)cudaStreamSynchronize(source.stream);
            } else {
                (void)cudaDeviceSynchronize();
            }
        }
        if (copies_start != nullptr) { (void)cudaEventDestroy(copies_start); }
        if (live_id != 0) {
            destroy_record(live_id, false);
        } else {
            arena_.free(block);
        }
        ++drops_;
        bump_version();
        std::fprintf(stderr, "[t8] capture drop reason=%s entries=%zu host_used=%zu\n",
                     drop.reason, records_.size(), host_footprint_bytes_);
        std::fflush(stderr);
        return false;
    } catch (...) {
        if (copies_launched) {
            if (source.stream != nullptr) {
                (void)cudaStreamSynchronize(source.stream);
            } else {
                (void)cudaDeviceSynchronize();
            }
        }
        if (copies_start != nullptr) { (void)cudaEventDestroy(copies_start); }
        if (live_id != 0) {
            destroy_record(live_id, false);
        } else {
            arena_.free(block);
        }
        throw;
    }
}

RamRestoredHost KVRamCache::unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target) {
    Record& record            = require(entry_id);
    std::fprintf(stderr, "[t8] restore id=%llu kv=%zu\n",
                 static_cast<unsigned long long>(entry_id), record.host_kv_bytes);
    std::fflush(stderr);
    wait_copies_on_stream(record, target.stream);
    const HeaderView header   = read_header(record.block, record.bytes);
    auto* raw                 = static_cast<std::uint8_t*>(record.block);
    const auto* fingerprint   = raw + kFixedHeader;
    InCursor fp{fingerprint, raw + header.header_bytes};
    if (target.text_cache == nullptr || target.text_pages.empty()) {
        throw std::invalid_argument("RAM restore target is incomplete");
    }
    verify_pool(fp, target.text_cache->page_pool(), header.text_plane_count, "text KV");
    if (header.backend_plane_count != 0) {
        if (target.backend_cache == nullptr || target.backend_pages.empty()) {
            throw std::logic_error("RAM restore is missing the backend pool");
        }
        verify_pool(fp, target.backend_cache->page_pool(), header.backend_plane_count,
                    "backend KV");
    }
    if (header.has_state_image) {
        // State-image record: the state half's geometry is the StateImage layout itself
        // (sections 4/5 are complete StateImage payloads); the per-type checks below only
        // apply to legacy per-type records.
        if (header.length[4] != 0 || header.length[5] != 0) {
            if (target.state_image == nullptr) {
                throw std::logic_error("RAM restore is missing the state image pool");
            }
            const std::size_t image_bytes = target.state_image->host_layout().image_bytes;
            if (header.length[4] != image_bytes) {
                throw std::logic_error("RAM entry state image geometry mismatch");
            }
            if (header.length[5] != 0 && header.length[5] != image_bytes) {
                throw std::logic_error("RAM entry state image checkpoint geometry mismatch");
            }
        }
    } else {
        if (target.tail_hidden != nullptr &&
            header.tail_hidden_bytes != target.tail_hidden->bytes()) {
            throw std::logic_error("RAM entry hidden geometry mismatch");
        }
    }

    begin_copies(record, target.stream);
    if (header.has_state_image) {
        if (header.length[4] != 0) {
            HostStateImageConstView current_view{reinterpret_cast<const std::byte*>(raw + header.offset[4]),
                                                 &target.state_image->host_layout()};
            target.state_image->copy_from_host(current_view, target.state_slot, target.stream);
        }
        if (header.length[5] != 0) {
            if (target.state_checkpoint_slot < 0) {
                throw std::logic_error("RAM restore is missing the state image checkpoint slot");
            }
            HostStateImageConstView checkpoint_view{reinterpret_cast<const std::byte*>(raw + header.offset[5]),
                                                    &target.state_image->host_layout()};
            target.state_image->copy_from_host(checkpoint_view, target.state_checkpoint_slot,
                                               target.stream);
        }
    }
    if (record.text_host_kv.valid()) {
        auto text_view = host_kv_arena_->view(record.text_host_kv);
        target.text_cache->page_pool().copy_from_host(text_view, target.text_pages, target.stream);
    }
    if (record.backend_host_kv.valid()) {
        if (target.backend_cache == nullptr || target.backend_pages.empty()) {
            throw std::logic_error("RAM restore is missing the backend page pool");
        }
        auto backend_view = host_kv_arena_->view(record.backend_host_kv);
        target.backend_cache->page_pool().copy_from_host(backend_view, target.backend_pages,
                                                         target.stream);
    }
    if (target.tail_hidden != nullptr && header.length[8] != 0) {
        CUDA_CHECK(cudaMemcpyAsync(target.tail_hidden->data, raw + header.offset[8],
                                   static_cast<std::size_t>(header.length[8]),
                                   cudaMemcpyHostToDevice, target.stream));
    }
    if (target.rewrite_checkpoint_hidden != nullptr && header.length[9] != 0) {
        if (header.length[9] != target.rewrite_checkpoint_hidden->bytes()) {
            throw std::logic_error("RAM entry rewrite-checkpoint hidden geometry mismatch");
        }
        CUDA_CHECK(cudaMemcpyAsync(target.rewrite_checkpoint_hidden->data, raw + header.offset[9],
                                   static_cast<std::size_t>(header.length[9]),
                                   cudaMemcpyHostToDevice, target.stream));
    }
    record_copies(record, target.stream);
    pending_load_id_ = entry_id;
    return host_from_header(record.block, header);
}

void KVRamCache::test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte) {
    Record& record            = require(entry_id);
    const HeaderView header   = read_header(record.block, record.bytes);
    auto* identity_bytes      = section_ptr(record.block, header, 1);
    ResidentPrefixIdentity identity;
    identity.unpack(identity_bytes, static_cast<std::size_t>(header.length[1]));
    identity.test_tamper_content_digest(0, byte);
    identity.pack(identity_bytes);
}

} // namespace ninfer::targets::qwen3_6::detail
