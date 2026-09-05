#pragma once

#include "targets/qwen3_6/impl/runtime/resource_projection.h"
#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <stdexcept>

// NINFER_QWEN36_RUNTIME_NS must be defined before this header (see instance.h), and program.h
// must be included first: the bodies below need the complete SequenceState/LongAnchorCheckpoint
// types that program.h defines.
//
// StateImage entitlement accounting shared by ProgramImplCore. Kept as inline free functions
// instead of private members so the entitlement invariants can be exercised by tests without a
// loaded model.

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

[[nodiscard]] inline std::uint32_t owned_checkpoint_references(const SequenceState& sequence,
                                                               StateImageHandle state) noexcept {
    std::uint32_t references = 0;
    if (sequence.rewrite_state && *sequence.rewrite_state == state) { ++references; }
    for (const LongAnchorCheckpoint& anchor : sequence.long_anchors) {
        if (anchor.state == state) { ++references; }
    }
    return references;
}

[[nodiscard]] inline bool state_exclusive_to_sequence(const StateImageStore& store,
                                                      const SequenceState& sequence,
                                                      StateImageHandle state) noexcept {
    if (!store.valid(state)) { return false; }
    return store.checkpoint_references(state) == owned_checkpoint_references(sequence, state);
}

[[nodiscard]] inline PhysicalResources
sequence_exclusive_state_resources(const StateImageStore& store, const SequenceState& sequence) {
    PhysicalResources out;
    std::array<StateImageHandle, 4> states{};
    std::uint32_t state_count = 0;
    const auto add_state      = [&](StateImageHandle handle) {
        if (!store.valid(handle)) {
            throw std::logic_error("sequence owner has a stale StateImage");
        }
        if (!state_exclusive_to_sequence(store, sequence, handle)) { return; }
        for (std::uint32_t index = 0; index < state_count; ++index) {
            if (states[index] == handle) { return; }
        }
        states[state_count++]                 = handle;
        const StateReplicaResidency residency = store.residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    };
    const bool has_read_state  = sequence.state.read.valid();
    const bool has_write_state = sequence.state.write.valid();
    if (has_read_state != has_write_state) {
        throw std::logic_error("sequence owner has a partial primary StateImage pair");
    }
    if (sequence.state.borrows_read() &&
        (!sequence.state.fork_pending || sequence.state.read == sequence.state.write)) {
        throw std::logic_error("sequence has an invalid borrowed StateImage source");
    }
    if (has_read_state) {
        if (!sequence.state.borrows_read() || sequence.state.read == sequence.state.write) {
            add_state(sequence.state.read);
        }
        add_state(sequence.state.write);
    }
    if (sequence.rewrite_state) { add_state(*sequence.rewrite_state); }
    if (sequence.reserved_state) { add_state(*sequence.reserved_state); }
    for (std::size_t anchor_index = 0; anchor_index < sequence.long_anchors.size();
         ++anchor_index) {
        const StateImageHandle handle = sequence.long_anchors[anchor_index].state;
        if (!store.valid(handle)) {
            throw std::logic_error("sequence owner has a stale long-anchor StateImage");
        }
        if (!state_exclusive_to_sequence(store, sequence, handle)) { continue; }
        bool seen = false;
        for (std::uint32_t index = 0; index < std::min<std::uint32_t>(state_count, states.size());
             ++index) {
            if (states[index] == handle) { seen = true; }
        }
        for (std::size_t prior = 0; !seen && prior < anchor_index; ++prior) {
            if (sequence.long_anchors[prior].state == handle) { seen = true; }
        }
        if (seen) { continue; }
        const StateReplicaResidency residency = store.residency(handle);
        if (residency == StateReplicaResidency::DeviceOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.device.state_slots;
        }
        if (residency == StateReplicaResidency::HostOnly ||
            residency == StateReplicaResidency::Both) {
            ++out.host.state_slots;
        }
    }
    return out;
}

// The planner budgeted `slots` exclusive device StateImage slots for `sequence`. The sequence's
// actually-owned exclusive slots must fit the budget exactly, and the only allowed growth is one
// fresh destination. Budgeting fewer slots than the sequence owns (the VISION-HIST gap) must be
// rejected here, not discovered later as a mid-flight accounting mismatch.
inline void reserve_state_entitlement(StateImageStore& store, SequenceState& sequence,
                                      std::uint32_t slots) {
    const std::uint32_t owned =
        sequence_exclusive_state_resources(store, sequence).device.state_slots;
    if (slots == 0 || owned > slots) {
        throw std::logic_error("sequence StateImage entitlement is inconsistent");
    }
    if (owned == slots) { return; }
    if (slots - owned != 1 || sequence.reserved_state) {
        throw std::logic_error("sequence StateImage reservation is not a single destination");
    }
    std::optional<StateImageHandle> reserved = store.reserve_destination();
    if (!reserved) { throw std::bad_alloc(); }
    sequence.reserved_state = *reserved;
    if (sequence_exclusive_state_resources(store, sequence).device.state_slots != slots) {
        throw std::logic_error("sequence StateImage entitlement did not materialize exactly");
    }
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
