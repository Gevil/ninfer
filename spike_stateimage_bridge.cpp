// v2-t8 spike (09-09): the substrate's STATE half (GDN + hidden + dflash_local) onto the baseline's
// state image. Validates the API mapping (per-TU syntax check) — confirms the substrate's
// RamCaptureSource state half maps onto the baseline's StateImageDevicePool (linear /
// continuation_hidden / dflash_local) + HostStatePool, with dflash_checkpoint as the one
// documented gap. Per advisory: keep the two mechanisms separate — the state image is slot-
// granular; the paged-KV host demotion (HostKVArena/HostKVExtentStore) is a distinct page-
// granular mechanism, not forced onto this abstraction.
#include "targets/qwen3_6/state_image.h"

#include <cuda_runtime_api.h>

#include <optional>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6;

// The state half of the substrate's RamCaptureSource: the fields that map onto the baseline's
// state image. The paged-KV half (text/backend PagedKVAllocation/Pool/Cache + residual_row) maps
// separately onto HostKVArena/HostKVExtentStore (a page-granular mechanism), NOT here.
struct SubstrateStateSlice {
    const LinearAttentionStatePool* gdn              = nullptr;  // == &pool.linear()
    const Tensor*                   tail_hidden      = nullptr;  // == &pool.continuation_hidden_slot(slot)
    const CyclicKVCache*            dflash_local     = nullptr;  // == pool.dflash_local()
    const CyclicKVCache*            dflash_checkpoint = nullptr; // GAP: baseline state image carries only dflash_local
    std::int32_t                    gdn_current_slot = -1;
};

// The state-image bridge, capture: the substrate's make_capture_header + pack for the state half,
// re-targeted onto StateImageDevicePool::copy_to_host + HostStatePool (the pinned host buffer).
void capture_state(const SubstrateStateSlice& src, const StateImageDevicePool& pool,
                   HostStatePool& host, cudaStream_t stream) {
    std::optional<HostStateSlotHandle> handle = host.allocate();
    if (!handle) { throw std::runtime_error("host state slot exhausted"); }
    pool.copy_to_host(src.gdn_current_slot, host.writable_view(*handle), stream);
    // The host slot now holds the GDN (linear conv + recurrent) + continuation hidden + dflash
    // local cyclic K/V for the slot — the baseline's StateImageHostLayout byte layout, not
    // gpillon's. The substrate's dflash_checkpoint has no baseline equivalent (documented gap).
    (void)src.gdn;
    (void)src.tail_hidden;
    (void)src.dflash_local;
    (void)src.dflash_checkpoint;
}

// The state-image bridge, restore: the substrate's unpack, re-targeted onto copy_from_host. The
// host view comes from a previously-captured HostStatePool slot (host.view(handle)).
void restore_state(const SubstrateStateSlice& src, StateImageDevicePool& pool,
                   HostStateImageConstView host_view, cudaStream_t stream) {
    pool.copy_from_host(host_view, src.gdn_current_slot, stream);
}
