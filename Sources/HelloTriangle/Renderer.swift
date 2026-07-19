import MetalKit

let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct Vertex {
      float2 position;
      float3 color;
    };

    vertex float4 vertex_main(uint vid [[vertex_id]], constant Vertex* vertices [[buffer(0)]]) {
        return float4(vertices[vid].position, 0.0, 1.0);
    }

    fragment float4 fragment_main() {
        return float4(1.0, 0.0, 0.0, 1.0);
    }
    """

struct Vertex {
    var positions: SIMD2<Float>
    var color: SIMD3<Float>
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    private let pipeline: MTLRenderPipelineState
    private let queue: MTLCommandQueue
    private let vertexBuffer: MTLBuffer

    init?(view: MTKView) {
        guard let device = view.device,
            let queue = device.makeCommandQueue()
        else { return nil }

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: metalSource, options: nil)
        } catch {
            print("Shader compile failed:\n\(error)")
            return nil
        }

        guard let vertex = library.makeFunction(name: "vertex_main"),
            let fragment = library.makeFunction(name: "fragment_main")
        else {
            print("Function not found in library")
            return nil
        }

        let verties = [
            Vertex(positions: [0.0, 0.8], color: [1, 0, 0]),
            Vertex(positions: [-0.8, -0.8], color: [0, 1, 0]),
            Vertex(positions: [0.8, 0.0], color: [0, 0, 1]),
        ]
        guard
            let vertexBuffer = device.makeBuffer(
                bytes: verties,
                length: MemoryLayout<Vertex>.stride * verties.count,
                options: .storageModeShared)
        else {
            return nil
        }
        self.vertexBuffer = vertexBuffer

        self.queue = queue

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertex
        desc.fragmentFunction = fragment
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat

        do {
            self.pipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            print("Pipeline creation failed:\n\(error)")
            return nil
        }
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
            let rpd = view.currentRenderPassDescriptor,
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeRenderCommandEncoder(descriptor: rpd)
        else { return }
        enc.setRenderPipelineState(pipeline)
        enc.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
