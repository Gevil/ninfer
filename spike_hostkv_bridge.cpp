// v2-t8 spike (09-09): the substrate's PAGED-KV half (text/backend paged-KV) onto the baseline's
// host-KV extent store (the host-KV page-replica demotion path). Validates the API mapping (per-TU
// syntax check) — confirms the substrate's whole-lane paged-KV checkpoint maps onto the baseline's
// HostKVExtentStore::prepare/publish (per the lane's page extents). Per advisory: this is a
// SEPARATE page-granular mechanism, not forced onto the slot-granular state image.
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"

#include <cuda_runtime_api.h>

#include <span>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::targets::qwen3_6::detail;

// The paged-KV half of the substrate's RamCaptureSource (text/backend paged-KV), re-targeted onto
// the baseline's HostKVExtentStore (the host-KV page-replica demotion path). The substrate's whole-
// lane checkpoint (text_pages/backend_pages) maps onto per-extent demotion: the policy bridge is
// the substrate's checkpoint policy vs the baseline's per-extent demotion.
struct SubstratePagedKVSlice {
    LogicalKVPageStore* text_store    = nullptr;  // -> the baseline's logical KV page store (text)
    LogicalKVPageStore* backend_store = nullptr;  // -> the baseline's logical KV page store (backend)
    std::span<const LogicalKVPageHandle> text_membership;    // the lane's text pages
    std::span<const LogicalKVPageHandle> backend_membership;  // the lane's backend pages
};

// The paged-KV bridge, capture: the substrate's whole-lane paged-KV checkpoint, re-targeted onto
// the baseline's HostKVExtentStore::prepare/publish (per the lane's page extents).
void capture_paged_kv(const SubstratePagedKVSlice& src, HostKVExtentStore& store,
                      cudaStream_t /*stream*/) {
    // The text extent: prepare -> device_sources (the DeviceKVPageHandle) + writable_view (the host
    // bytes) -> (a DMA copy, elided) -> publish (the published host-KV extent).
    std::optional<HostKVExtentReservation> text_res =
        store.prepare(*src.text_store, src.text_membership);
    if (text_res) {
        std::vector<DeviceKVPageHandle> sources = store.device_sources(*text_res);
        auto host = store.writable_view(*text_res);
        auto cap  = store.publish(std::move(*text_res));
        (void)sources;
        (void)host;
        (void)cap;
    }
    // The backend extent (same, for the backend pages).
    std::optional<HostKVExtentReservation> backend_res =
        store.prepare(*src.backend_store, src.backend_membership);
    if (backend_res) {
        auto cap = store.publish(std::move(*backend_res));
        (void)cap;
    }
}
