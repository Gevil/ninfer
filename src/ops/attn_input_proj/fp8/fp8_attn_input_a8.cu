#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"
#include "ops/linear/fp8/fp8_a8_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8AttnInputGeometry;

template <class Schedule, bool FullTokens>
void launch_mma(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                Fp8A8Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    static_assert((kFp8AttnInputQueryRows % Schedule::kBlockRows) == 0);
    static_assert((kFp8AttnInputKeyRows % Schedule::kBlockRows) == 0);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int blocks        = kRowTiles * token_tiles;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };

    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    fp8_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            workspace.codes, workspace.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), tokens, Fp8IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void run(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
         Fp8A8Workspace workspace, int tokens, cudaStream_t stream) {
    if (tokens % Schedule::kBlockTokens == 0)
        launch_mma<Schedule, true>(weight, q, gate, k, v, workspace, tokens, stream);
    else
        launch_mma<Schedule, false>(weight, q, gate, k, v, workspace, tokens, stream);
}
} // namespace

void fp8_attn_input_a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    using TmaSchedule = typename Fp8LinearA8TmaSchedule<Geometry>::Type;
    if (fp8_a8_tma_applies<Geometry, TmaSchedule, Schedule>(x.ne[1], workspace.codes,
                                                            weight.qdata)) {
        const Fp8AttentionInputOutput output{
            static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data),
            static_cast<__nv_bfloat16*>(v.data),
        };
        fp8_a8_tma_launch<Geometry, TmaSchedule>(workspace.codes, workspace.scales,
                                                 static_cast<const std::uint8_t*>(weight.qdata),
                                                 static_cast<const __nv_bfloat16*>(weight.scales),
                                                 x.ne[1], Fp8IdentityEpilogue{}, output, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if ((x.ne[1] % Schedule::kBlockTokens) == 0) {
        launch_mma<true>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    } else {
        launch_mma<false>(weight, q, gate, k, v, workspace, x.ne[1], stream);
    }
}
} // namespace ninfer::ops::detail
