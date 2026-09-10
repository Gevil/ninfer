#pragma once

#include "core/arena.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"
#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "targets/qwen3_6/impl/runtime/kv_ram_snapshot.h"
#include <ninfer/targets/qwen3_6/state_image.h>
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

namespace ninfer::targets::qwen3_6::detail {

// What produced a capture, for eviction/accounting policy that needs to tell captures apart
// without inferring it from other fields. Terminal is a settled lane retired into the cache;
// ActiveSibling and SharedBoundary are the two multi_claim capture sites that existed before the
// dynamic boundary; DynamicBoundary is the LCP-driven capture chosen by admission's boundary
// policy (see runtime::choose_boundary_capture) and is the only kind subject to a live-count cap.
enum class RamCaptureKind : std::uint8_t {
    Terminal,
    ActiveSibling,
    SharedBoundary,
    DynamicBoundary,
};

// Admission-selected captures must preserve all existing records. Terminal retention can retain
// the normal cache policy, which is allowed to reclaim cold unclaimed records for progress.
enum class RamCapturePolicy : std::uint8_t {
    AllowEviction,
    PreserveExisting,
};

struct RamCaptureSource {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;

    std::span<const TokenId> ledger;
    const ResidentPrefixIdentity* identity = nullptr;
    PrefixHash128 hash_f{};
    PrefixHash128 hash_c{};
    bool hash_c_valid = false;

    // Exact physical image extents: the device pages whose bytes are captured into the host KV
    // arena image (see KVRamCache::host_kv_arena_). Capturing a reusable prefix must not
    // serialize pages that were materialized speculatively for the remainder of the prompt.
    std::span<const DeviceKVPageHandle> text_pages    = {};
    std::span<const DeviceKVPageHandle> backend_pages = {};

    // The caches themselves: source of the device page pool (geometry + stream-async host
    // copy) for the paged image captured above.
    const qwen3_6::PagedKVCache* text_cache    = nullptr;
    const qwen3_6::PagedKVCache* backend_cache = nullptr;

    const Tensor* tail_hidden                = nullptr;
    const Tensor* rewrite_checkpoint_hidden  = nullptr;

    // The baseline's state-image pool: each absolute slot holds the complete continuation state
    // (GDN linear conv+recurrent, continuation hidden, DFlash local cyclic K/V). When set, the
    // state half is captured/restored as complete StateImage payloads (capture block sections
    // 4/5, laid out per StateImageHostLayout) via copy_to_host/copy_from_host landing directly
    // in the capture block -- no per-type host-image pack, no pinned-pool round trip. The
    // per-type state fields above (gdn/tail_hidden/dflash_* + slots) are then ignored for the
    // state half. Null state_image = legacy per-type state half (or KV-only capture).
    const StateImageDevicePool* state_image         = nullptr;
    std::int32_t state_slot            = 0;    // absolute slot captured into section 4
    std::int32_t state_checkpoint_slot = -1;   // absolute slot captured into section 5

    cudaStream_t stream = nullptr;

    RequestClass owner_class = RequestClass::Agents;

    // True for a speculative snapshot of a lane that is still actively serving its own request
    // (captured right as its prefill completes, before decode -- see
    // ProgramImplCore::capture_active_lane_for_siblings), taken on the chance that another
    // pending request shares enough of its leading prompt to be worth restoring from it. Unlike
    // an ordinary terminal-lane capture, more than one sibling may legitimately want to restore
    // from the same entry, so it must not be erased after the first restore.
    bool multi_claim = false;

    RamCaptureKind capture_kind = RamCaptureKind::Terminal;
};

struct RamRestoreTarget {
    // The destination device pages the record's host image is restored into (see
    // RamCaptureSource::text_pages).
    std::span<const DeviceKVPageHandle> text_pages    = {};
    std::span<const DeviceKVPageHandle> backend_pages = {};

    // The caches themselves: destination of the stream-async host->device page copy.
    qwen3_6::PagedKVCache* text_cache    = nullptr;
    qwen3_6::PagedKVCache* backend_cache = nullptr;

    Tensor* tail_hidden               = nullptr;
    Tensor* rewrite_checkpoint_hidden = nullptr;


    // State-image restore target: the destination pool + slots the record's state-image
    // sections (4/5) are restored into via StateImageDevicePool::copy_from_host.
    StateImageDevicePool* state_image         = nullptr;
    std::int32_t state_slot            = 0;
    std::int32_t state_checkpoint_slot = -1;

    cudaStream_t stream = nullptr;
};

struct RamRestoredHost {
    std::uint32_t execution_frontier      = 0;
    std::uint32_t ledger_frontier         = 0;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    bool tail_hidden_valid                = false;
    bool rewrite_valid                    = false;
    RewriteCheckpointKind rewrite_kind    = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier        = 0;
    bool backend_image_present            = false;
    std::vector<TokenId> ledger;
    ResidentPrefixIdentity identity;
};

struct RamMatch {
    std::uint64_t entry_id     = 0;
    PrefixReusePath reuse      = PrefixReusePath::FullReset;
    std::uint32_t reuse_base   = 0;
};

class KVRamCache {
public:
    KVRamCache(std::size_t capacity_bytes, HostKVArena& host_kv_arena);
    ~KVRamCache();

    KVRamCache(const KVRamCache&)            = delete;
    KVRamCache& operator=(const KVRamCache&) = delete;
    KVRamCache(KVRamCache&&)                 = delete;
    KVRamCache& operator=(KVRamCache&&)      = delete;

    [[nodiscard]] std::optional<RamMatch> plan_match(const PreparedPromptData& prompt,
                                                     std::span<const PrefixHash128> hash_chain);

    void claim(std::uint64_t entry_id);
    void release(std::uint64_t entry_id);
    void consume(std::uint64_t entry_id);

    // Exact record footprint produced by capture(source), including every serialized section and
    // alignment. This is the single sizing authority used by target admission preflight.
    [[nodiscard]] std::size_t capture_bytes(const RamCaptureSource& source) const;
    // True only when this source can be captured now without reclaiming a record, including the
    // DynamicBoundary live-count policy. The actual PreserveExisting capture repeats the check.
    [[nodiscard]] bool can_capture_without_eviction(const RamCaptureSource& source);
    bool capture(const RamCaptureSource& source,
                 RamCapturePolicy policy = RamCapturePolicy::AllowEviction);
    RamRestoredHost unpack_device(std::uint64_t entry_id, const RamRestoreTarget& target);

    [[nodiscard]] RamRestoredHost load_host(std::uint64_t entry_id) const;

    [[nodiscard]] KvRamSnapshot snapshot() const noexcept;
    KvRamCopySeconds harvest_copy_seconds();
    [[nodiscard]] std::uint64_t index_version() const noexcept { return index_version_; }
    [[nodiscard]] std::uint64_t exact_comparisons() const noexcept { return exact_comparisons_; }

    void test_tamper_identity_digest(std::uint64_t entry_id, std::uint8_t byte);

private:
    enum class Section : std::uint8_t {
        Ledger = 0,
        Identity,
        TextKv,
        BackendKv,
        GdnConvCurrent,
        GdnConvCheckpoint,
        GdnRecurrentCurrent,
        GdnRecurrentCheckpoint,
        TailHidden,
        RewriteCheckpointHidden,
        DflashLocal,
        DflashRewriteCheckpoint,
        // The hyperquant exact-key side store for the captured slot row: one image per side plane
        // across every layer, plus the row's recent-ring validity words. Absent (length 0) for
        // every KV dtype that keeps no side store.
        TextResidualK,
        TextResidualV,
        TextRingValid,
        BackendResidualK,
        BackendResidualV,
        BackendRingValid,
        Count
    };

    struct Record {
        std::uint64_t id               = 0;
        PrefixHash128 hash_f{};
        PrefixHash128 hash_c{};
        bool hash_c_valid              = false;
        std::uint32_t execution_frontier = 0;
        std::uint32_t checkpoint_frontier = 0;
        bool checkpoint_valid          = false;
        PrefixReusePath checkpoint_path = PrefixReusePath::RestoreTurnCheckpoint;
        void* block                    = nullptr;
        std::size_t bytes              = 0;
        // Paged-KV image bytes held in the shared host KV arena (text + backend spans).
        // Charged against the tier budget by capture() alongside the flat block bytes.
        std::size_t host_kv_bytes      = 0;
        // The record's paged-KV image lives in the shared host KV arena, not in the flat block
        // (sections 2/3 are unused): one arena allocation per non-empty page span, released
        // with the record -- or moved into the record's RetiredCopy until the in-flight copy
        // lands.
        HostKVAllocation text_host_kv;
        HostKVAllocation backend_host_kv;
        // Outstanding claims. An ordinary record admits exactly one claimant at a time (claim()
        // rejects a second), preserving the original exclusive capture/match/consume lifecycle.
        // A multi_claim record admits several concurrently, so a burst of siblings all restore
        // from one snapshot instead of each capturing its own duplicate of the same lane.
        std::uint32_t claims           = 0;
        bool copies_timed              = false;
        cudaEvent_t copies_start       = nullptr;
        cudaEvent_t copies_done        = nullptr;
        // Lineage identity (hash of the leading tokens, typically the system prompt + tool
        // schema) and whether that lineage has previously produced a cache hit. Entries whose
        // lineage has demonstrated reuse are evicted only after all non-protected entries are
        // exhausted, so short one-shot captures (classifier calls, etc.) churn out first instead
        // of displacing checkpoints from conversations that keep coming back.
        std::uint64_t origin_hash      = 0;
        bool protected_tier            = false;
        // See RamCaptureSource::multi_claim. consume() releases the claim instead of erasing
        // the record for entries captured this way, so a second/third sibling can independently
        // claim/restore the same entry; it then ages out through ordinary eviction like any
        // other record, no special-cased cleanup required.
        bool multi_claim               = false;
        RequestClass owner_class = RequestClass::Agents;
        RamCaptureKind capture_kind    = RamCaptureKind::Terminal;
        // Longest-common-prefix ladder sampled at capture: ascending frontier points and the
        // rolling prefix hash at each (in-memory only, not serialized). A candidate chain
        // matching to a point proves agreement over [0:point]; on this hybrid state model an
        // in-flight point is diagnostic only -- reusable points must carry a state snapshot,
        // which only the frontier and the checkpoint frontiers do.
        std::vector<std::uint32_t> ladder_points;
        std::vector<PrefixHash128> ladder_hashes;
    };

    struct Layout {
        std::size_t header_bytes                               = 0;
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> offset{};
        std::array<std::size_t, static_cast<std::size_t>(Section::Count)> length{};
        std::size_t entry_bytes                                = 0;
    };

    [[nodiscard]] Record& require(std::uint64_t entry_id);
    [[nodiscard]] const Record& require(std::uint64_t entry_id) const;
    [[nodiscard]] bool lineage_is_hot(std::uint64_t origin_hash) const;
    void note_lineage_hit(std::uint64_t origin_hash);
    void evict_unpinned();
    void destroy_record(std::uint64_t entry_id, bool count_eviction);
    // Moves entry_id to the back of fifo_ on a demonstrated hit, so eviction (which always walks
    // fifo_ front-to-back within a rank) reclaims the coldest record of that rank/kind rather than
    // simply the oldest by capture time.
    void touch(std::uint64_t entry_id);
    // Enforces the live-count cap on RamCaptureKind::DynamicBoundary records before a new one is
    // admitted, evicting the coldest unclaimed one if the cap is already met. Returns false only
    // when the cap is met and every existing DynamicBoundary record is claimed -- the caller must
    // then fail the capture rather than force an eviction of state still in use.
    bool make_room_for_dynamic_boundary();
    [[nodiscard]] bool has_room_for_dynamic_boundary_without_eviction() const noexcept;
    void begin_copies(Record& record, cudaStream_t stream);
    void record_copies(Record& record, cudaStream_t stream);
    void wait_copies(Record& record);
    void wait_copies_on_stream(Record& record, cudaStream_t stream);
    double harvest_record(Record& record);
    void retire_record(Record& record);
    void reap_retired(bool block);
    void bump_version() noexcept { ++index_version_; }

    struct RetiredCopy {
        void* block             = nullptr;
        cudaEvent_t copies_done = nullptr;
        HostKVAllocation text_host_kv;
        HostKVAllocation backend_host_kv;
    };

    HostPinnedArena arena_;
    // The paged-KV image half is stored in the lane's shared host KV arena (the same pool the
    // extent store demotes into); the arena outlives this cache.
    HostKVArena* host_kv_arena_ = nullptr;
    // Live host-RAM footprint of the tier: sum over live records of (flat block bytes +
    // paged-KV image bytes). capture() enforces capacity_bytes against this total, so
    // --kv-ram-capacity-mib bounds the tier's entire host commitment, not just the flat
    // capture blocks. Charged on capture, discharged on retire/destroy.
    std::size_t host_footprint_bytes_ = 0;
    // Paged-KV image half of the footprint; reported as KvRamSnapshot::kv_image_bytes.
    std::size_t host_kv_bytes_used_   = 0;
    std::deque<std::uint64_t> fifo_;
    std::unordered_map<std::uint64_t, Record> records_;
    // Bounded record of which content lineages (leading-token hash) have previously produced a
    // cache hit, so a new capture for the same conversation can start out protected. Capped and
    // FIFO-evicted independently of the KV entries themselves -- this is a small hint table, not
    // a source of truth.
    std::unordered_map<std::uint64_t, std::uint32_t> lineage_hits_;
    std::deque<std::uint64_t> lineage_order_;
    std::vector<RetiredCopy> retired_;
    std::vector<std::uint64_t> pending_save_ids_;
    std::optional<std::uint64_t> pending_load_id_;
    std::uint64_t next_id_           = 1;
    std::uint64_t index_version_     = 1;
    std::uint64_t captures_          = 0;
    std::uint64_t restores_          = 0;
    std::uint64_t evictions_         = 0;
    std::uint64_t drops_             = 0;
    std::uint64_t exact_comparisons_ = 0;
    double save_seconds_             = 0;
    double load_seconds_             = 0;
    double orphaned_save_seconds_    = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
