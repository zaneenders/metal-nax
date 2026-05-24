//  nax_shader.metal
//  Basic Metal 4 compute shader leveraging NAX (Neural Acceleration eXtension)
//  via Metal Performance Primitives (MPP) and bfloat16 tensor ops.
//
//  Requires: Apple M4/M5 or later (this machine has Apple M5)
//  Compile with: xcrun -sdk macosx metal -c nax_shader.metal -o build/nax_shader.air
//                xcrun -sdk macosx metallib build/nax_shader.air -o build/nax_shader.metallib

#include <metal_stdlib>
#include <MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h>

using namespace metal;
using namespace mpp::tensor_ops;

// MARK: - Simple fused kernel: C = relu(A * B + bias)
//
// This kernel demonstrates NAX by:
//  1. Using bfloat16 (the native NAX data type)
//  2. Using MPP matmul2d which internally maps to NAX matrix-multiply instructions
//  3. Keeping the matmul result in a cooperative_tensor (in-register) to fuse
//     the bias-add and ReLU without round-tripping to memory.
//
//  Dispatch: threadgroups of (simdgroupWidth * 4, 1, 1) threads.
//            Each threadgroup cooperatively computes a 64x32 tile of output.

kernel void naxFusedMatmulBiasRelu(
    // Input tensors in device memory
    tensor<device bfloat, dextents<int32_t, 2>>  A,   // M x K
    tensor<device bfloat, dextents<int32_t, 2>>  B,   // K x N
    tensor<device bfloat, dextents<int32_t, 2>>  bias, // 1 x N (broadcasted)
    tensor<device bfloat, dextents<int32_t, 2>>  C,   // M x N (output)

    // Dimensions
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],

    // Grid position
    uint2 tgid [[threadgroup_position_in_grid]]
)
{
    // Descriptor: 64x32 output tile, dynamic K, no transposes, relaxed precision
    // relaxed_precision=true allows the implementation to use NAX fast-paths
    // that may trade a tiny bit of accuracy for significant speedup.
    constexpr auto descriptor = matmul2d_descriptor(
        64,   // m: rows in output tile
        32,   // n: cols in output tile
        static_cast<int>(dynamic_extent), // k: dynamic inner dim
        false, // transpose_left
        false, // transpose_right
        true   // relaxed_precision (enable NAX fast paths)
    );

    // Execution scope: 4 SIMD groups per threadgroup working cooperatively.
    // All threads in these 4 SIMD groups must enter the .run() call together.
    matmul2d<descriptor, execution_simdgroups<4>> matmulOp;

    // Slice the global tensors for this threadgroup's tile.
    // A is M x K, we take a vertical slice of 64 rows starting at tgid.y*64.
    // B is K x N, we take a horizontal slice of 32 cols starting at tgid.x*32.
    // C is M x N, same tile as the output.
    auto mA = A.slice(0, int(tgid.y) * 64);
    auto mB = B.slice(int(tgid.x) * 32, 0);
    auto mC = C.slice(int(tgid.x) * 32, int(tgid.y) * 64);

    // -------------------------------------------------------------------------
    // 1. Allocate a cooperative destination tensor in thread-private registers.
    //    This avoids writing the raw matmul result to device memory.
    // -------------------------------------------------------------------------
    auto cT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), bfloat>();

    // Initialise cooperative tensor to zero (required before accumulate).
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = bfloat(0.0f);
        }
    }

    // -------------------------------------------------------------------------
    // 2. Run the matrix multiplication directly into the cooperative tensor.
    //    MPP will dispatch this to NAX when bfloat + relaxed_precision is used.
    // -------------------------------------------------------------------------
    matmulOp.run(mA, mB, cT);

    // -------------------------------------------------------------------------
    // 3. Load bias into a second cooperative tensor with the same layout.
    // -------------------------------------------------------------------------
    auto biasT = matmulOp.get_destination_cooperative_tensor<
        decltype(mA), decltype(mB), bfloat>();

    // Load bias into cooperative tensor (layout matches destination tile).
    biasT.load(bias);

    // -------------------------------------------------------------------------
    // 4. Fused element-wise ops: add bias, then ReLU (max(0, x)).
    //    Everything stays in registers — no extra memory traffic.
    // -------------------------------------------------------------------------
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            bfloat val = cT[i] + biasT[i];
            cT[i] = select(bfloat(0.0f), val, val > bfloat(0.0f)); // ReLU
        }
    }

    // -------------------------------------------------------------------------
    // 5. Store the final result back to device memory.
    // -------------------------------------------------------------------------
    cT.store(mC);
}


// MARK: - Simple bfloat copy / type-check kernel
//
// A minimal kernel that just exercises the bfloat type and NAX presence.
// Useful for verifying the toolchain recognises bfloat on this M5 machine.

kernel void naxBfloatCopy(
    const device bfloat* src  [[buffer(0)]],
    device       bfloat* dst  [[buffer(1)]],
    constant     uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
)
{
    if (gid >= count) return;

    // Simple vectorised load/store using bfloat4 if available.
    // Falls back to scalar on older compilers.
#if __HAVE_BFLOAT__
    dst[gid] = src[gid] * bfloat(2.0f);
#else
    // Should never hit on M5, but keeps the code portable.
    dst[gid] = src[gid];
#endif
}
