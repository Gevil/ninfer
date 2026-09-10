#pragma once

#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_6::detail {

struct KvRamSnapshot {
    std::size_t capacity_bytes  = 0;
    // Sum of indexed Record::bytes, including a claimed-but-not-consumed pin.
    // Retired copy blocks still occupying the pin are excluded until reap.
    std::size_t used_bytes      = 0;
    // Sum of the live records' paged-KV image bytes (held in the lane's shared host KV arena,
    // not in the flat capture blocks). Together with used_bytes this is the tier's total
    // host-RAM footprint, which capture() enforces against capacity_bytes.
    std::size_t kv_image_bytes  = 0;
    std::size_t entry_count     = 0;
    std::uint64_t captures      = 0;
    std::uint64_t restores      = 0;
    std::uint64_t evictions     = 0;
    std::uint64_t drops         = 0;
    // Host-side time spent serializing captures (device->host copies + ledger/identity pack),
    // harvested by the stats publisher.
    double save_seconds         = 0;
    // Host-side time spent restoring records (host->device copies), harvested the same way.
    double load_seconds          = 0;
};

// Accumulated copy-side seconds returned to the stats publisher between harvests.
struct KvRamCopySeconds {
    double save = 0;
    double load = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
