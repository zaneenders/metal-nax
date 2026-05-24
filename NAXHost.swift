import Foundation
import Metal
import MetalPerformanceShaders

final class NAXContext {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init?() {
        // 1. Pick the default Metal device (Apple M5 GPU).
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No Metal device found")
            return nil
        }
        self.device = device

        // 2. Verify bfloat16 / NAX support.
        //    NAX is an M5-only GPU hardware feature (Neural Accelerator inside
        //    each GPU core). Metal 4 itself runs on M1+, but NAX requires M5.
        //    M5 maps to MTLGPUFamily.apple10. We also check .metal4 if available.
        let supportsNAX: Bool
        if #available(macOS 26, iOS 26, tvOS 26, *) {
            supportsNAX = device.supportsFamily(.metal4)
        } else {
            supportsNAX = device.supportsFamily(.apple10)
        }
        print("Device: \(device.name)")
        print("Supports NAX (Metal 4): \(supportsNAX)")

        guard supportsNAX else {
            print("NAX requires Apple M5 or later.")
            return nil
        }

        // 3. Load the pre-compiled metallib or compile from source.
        //    For a quick test you can compile the .metal file manually:
        //    xcrun -sdk macosx metal    nax_shader.metal -o nax_shader.air
        //    xcrun -sdk macosx metallib nax_shader.air  -o nax_shader.metallib
        let libraryURL = URL(fileURLWithPath: "build/nax_shader.metallib")
        guard let library = try? device.makeLibrary(URL: libraryURL) else {
            print("Failed to load Metal library from \(libraryURL.path).")
            return nil
        }

        guard let fn = library.makeFunction(name: "naxFusedMatmulBiasRelu") else {
            print("Kernel 'naxFusedMatmulBiasRelu' not found in library")
            return nil
        }

        // 4. Create compute pipeline.
        //    MTLComputePipelineDescriptor lets you opt into MPP tensor ops.
        let desc = MTLComputePipelineDescriptor()
        desc.computeFunction = fn
        desc.supportIndirectCommandBuffers = false

        do {
            self.pipeline = try device.makeComputePipelineState(
                descriptor: desc, options: [], reflection: nil)
        } catch {
            print("Pipeline creation failed: \(error)")
            return nil
        }

        guard let queue = device.makeCommandQueue() else { return nil }
        self.commandQueue = queue
    }

    // MARK: - Run fused Matmul + Bias + ReLU

    func runFusedMatmulBiasRelu(
        M: Int, N: Int, K: Int
    ) {
        // Allocate bfloat16 tensors.
        // Note: MTLPixelFormat.bgr10a2Unorm etc are NOT bfloat; for raw buffers
        // we use .bfloat as the MTLDataType when creating MTLBuffers via MPS.
        // Here we simply allocate raw buffers and cast to bfloat in the shader.

        let aSize = M * K * MemoryLayout<UInt16>.size  // bfloat is 16-bit
        let bSize = K * N * MemoryLayout<UInt16>.size
        let cSize = M * N * MemoryLayout<UInt16>.size
        let biasSize = N * MemoryLayout<UInt16>.size

        guard let bufferA = device.makeBuffer(length: aSize, options: .storageModeShared),
            let bufferB = device.makeBuffer(length: bSize, options: .storageModeShared),
            let bufferBias = device.makeBuffer(length: biasSize, options: .storageModeShared),
            let bufferC = device.makeBuffer(length: cSize, options: .storageModeShared)
        else {
            print("Buffer allocation failed")
            return
        }

        // Fill with dummy data (as UInt16 bit patterns).
        // In production you would convert Float32 -> bfloat16 before upload.
        fillRandomBFloat(buffer: bufferA, count: M * K)
        fillRandomBFloat(buffer: bufferB, count: K * N)
        fillRandomBFloat(buffer: bufferBias, count: N)

        // Create tensor descriptors for MPP.
        // MPP tensor ops use the tensor<T, dextents<...>> type in the shader,
        // but on the CPU side we just pass the buffer and dimensions.
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
            let encoder = cmdBuf.makeComputeCommandEncoder()
        else {
            print("Command buffer/encoder creation failed")
            return
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferBias, offset: 0, index: 2)
        encoder.setBuffer(bufferC, offset: 0, index: 3)

        var mVal = UInt32(M)
        var nVal = UInt32(N)
        var kVal = UInt32(K)
        encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 6)

        // Threadgroup size: 4 SIMD groups wide.
        // The shader uses execution_simdgroups<4>, so we must match that.
        let simdgroupWidth = pipeline.threadExecutionWidth  // typically 32 on Apple Silicon
        let threadsPerThreadgroup = MTLSize(
            width: simdgroupWidth * 4,
            height: 1,
            depth: 1
        )

        // Grid size: enough threadgroups to cover the M x N output matrix.
        let threadgroups = MTLSize(
            width: (N + 31) / 32,  // 32 cols per threadgroup
            height: (M + 63) / 64,  // 64 rows per threadgroup
            depth: 1
        )

        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        // Add completion handler to verify execution.
        cmdBuf.addCompletedHandler { _ in
            print("NAX kernel completed: \(M)x\(N) output computed.")
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    // MARK: - Helpers

    private func fillRandomBFloat(buffer: MTLBuffer, count: Int) {
        // bfloat16 has the same exponent as float32 but only 7 mantissa bits.
        // Quick conversion: truncate the lower 16 bits of a float32.
        let ptr = buffer.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            var f = Float.random(in: -1.0...1.0)
            var u: UInt32 = 0
            memcpy(&u, &f, 4)
            ptr[i] = UInt16(u >> 16)  // upper 16 bits = bfloat16
        }
    }
}

// MARK: - Entry point (for command-line or playground)
if let ctx = NAXContext() {
    // Small example: 128 x 64 matmul
    ctx.runFusedMatmulBiasRelu(M: 128, N: 64, K: 128)
}
