#include "core/device.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime

#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/state_image_entitlement.h"

#include <ninfer/targets/qwen3_6/state_image.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>

namespace {

namespace q36  = ninfer::targets::qwen3_6;
using q36::detail::StateImageStore;
using namespace ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

struct PlannedPool {
    q36::StateImageDeviceLayout layout;
    std::size_t bytes = 0;
};

PlannedPool plan_pool(std::int32_t slots = 4) {
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
    ninfer::LayoutBuilder builder;
    q36::StateImageDeviceLayout layout = q36::plan_state_image_device_pool(builder, spec);
    return {.layout = std::move(layout), .bytes = builder.finish(256)};
}

void expect_entitlement_rejected(std::string_view message_fragment, StateImageStore& store,
                                 SequenceState& sequence, std::uint32_t slots) {
    try {
        reserve_state_entitlement(store, sequence, slots);
    } catch (const std::logic_error& error) {
        expect(std::string(error.what()).find(message_fragment) != std::string::npos,
               "reservation rejection message for slots=" + std::to_string(slots));
        return;
    }
    expect(false, "reservation must reject slots=" + std::to_string(slots));
}

} // namespace

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err) || count == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    CUDA_CHECK(count_err);

    PlannedPool planned = plan_pool(4);
    ninfer::DeviceArena arena(planned.bytes);
    q36::StateImageDevicePool pool({arena.base(), arena.capacity()}, planned.layout);

    // No host pool: every image is DeviceOnly, so all accounting is device-residency only.
    StateImageStore store(pool, nullptr, static_cast<std::uint32_t>(pool.slot_count()));

    // h1: the sequence's primary pair (shared read+write), zero external references.
    const auto h1 = *store.reserve_destination();
    // h2: a long-anchor checkpoint exclusively owned by the sequence (one reference of its own).
    const auto h2 = *store.reserve_destination();
    store.activate_reset(h2);
    store.freeze(h2);
    store.retain_checkpoint_reference(h2);
    // h3: a checkpoint whose reference belongs to ANOTHER sequence (borrowed read source).
    const auto h3 = *store.reserve_destination();
    store.activate_reset(h3);
    store.freeze(h3);
    store.retain_checkpoint_reference(h3);

    SequenceState sequence{};
    sequence.state.read         = h3; // borrowed read: externally owned, not exclusive here
    sequence.state.write        = h1;
    sequence.long_anchors.push_back({.state = h2, .frontier = 1, .ordinal = 0});

    expect(!state_exclusive_to_sequence(store, sequence, h3), "borrowed read is not exclusive");
    expect(state_exclusive_to_sequence(store, sequence, h1), "primary pair is exclusive");
    expect(state_exclusive_to_sequence(store, sequence, h2), "long anchor is exclusive");

    const auto owned_resources = sequence_exclusive_state_resources(store, sequence);
    expect(owned_resources.device.state_slots == 2,
           "primary pair + exclusive anchor count as two device slots");
    expect(owned_resources.host.state_slots == 0, "no host residency without a host pool");

    SequenceState other{};
    other.long_anchors.push_back({.state = h3, .frontier = 1, .ordinal = 0});
    expect(state_exclusive_to_sequence(store, other, h3), "same image is exclusive to its owner");
    expect(sequence_exclusive_state_resources(store, other).device.state_slots == 1,
           "external owner counts only its own reference");

    // --- reserve_state_entitlement contract ---

    // slots == owned: nothing to reserve.
    reserve_state_entitlement(store, sequence, 2);
    expect(!sequence.reserved_state.has_value(), "exact entitlement reserves no destination");

    // The planner gap (VISION-HIST): budgeting fewer slots than the sequence actually owns must
    // be rejected, not silently dropped.
    expect_entitlement_rejected("entitlement is inconsistent", store, sequence, 1);
    expect_entitlement_rejected("not a single destination", store, sequence, 4);
    expect_entitlement_rejected("entitlement is inconsistent", store, sequence, 0);

    // owned + 1: the reservation materializes exactly one destination.
    const std::uint32_t occupied_before = store.occupied();
    reserve_state_entitlement(store, sequence, 3);
    expect(sequence.reserved_state.has_value(), "single extra slot reserves a destination");
    expect(sequence_exclusive_state_resources(store, sequence).device.state_slots == 3,
           "the reservation materialized exactly");
    expect(store.occupied() == occupied_before + 1, "the destination consumed one logical image");

    return failures == 0 ? 0 : 1;
}
