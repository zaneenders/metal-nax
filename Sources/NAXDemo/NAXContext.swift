import Foundation
import Metal
import MetalPerformanceShaders
import NAXShaders

final class NAXContext {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let pipeline: MTLComputePipelineState

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("No Metal device found")
            return nil
        }
        self.device = device

        // Verify bfloat16 / NAX support.
        // NAX is an M5-only GPU hardware feature (Neural Accelerator inside
        // each GPU core). Metal 4 itself runs on M1+, but NAX requires M5.
        // M5 maps to MTLGPUFamily.apple10. We also check .metal4 if available.
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

        let libraryURL = NAXShaderLibrary.metallibURL
        guard let library = try? device.makeLibrary(URL: libraryURL) else {
            print("Failed to load Metal library from \(libraryURL.path).")
            return nil
        }

        guard let fn = library.makeFunction(name: "naxFusedMatmulBiasRelu") else {
            print("Kernel 'naxFusedMatmulBiasRelu' not found in library")
            return nil
        }

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

    func runFusedMatmulBiasRelu(
        M: Int, N: Int, K: Int
    ) {
        let aSize = M * K * MemoryLayout<UInt16>.size
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

        fillRandomBFloat(buffer: bufferA, count: M * K)
        fillRandomBFloat(buffer: bufferB, count: K * N)
        fillRandomBFloat(buffer: bufferBias, count: N)

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

        let simdgroupWidth = pipeline.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(
            width: simdgroupWidth * 4,
            height: 1,
            depth: 1
        )

        let threadgroups = MTLSize(
            width: (N + 31) / 32,
            height: (M + 63) / 64,
            depth: 1
        )

        encoder.dispatchThreadgroups(
            threadgroups,
            threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        cmdBuf.addCompletedHandler { _ in
            print("NAX kernel completed: \(M)x\(N) output computed.")
        }

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
    }

    private func fillRandomBFloat(buffer: MTLBuffer, count: Int) {
        let ptr = buffer.contents().assumingMemoryBound(to: UInt16.self)
        for i in 0..<count {
            var f = Float.random(in: -1.0...1.0)
            var u: UInt32 = 0
            memcpy(&u, &f, 4)
            ptr[i] = UInt16(u >> 16)
        }
    }
}
