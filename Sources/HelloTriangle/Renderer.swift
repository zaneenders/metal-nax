import MetalKit

let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    vertex float4 vertex_main(uint vid [[vertex_id]]) {
        const float2 positions[3] = {
            float2( 0.0,  0.8),
            float2(-0.8, -0.8),
            float2( 0.8, -0.8)
        };
        return float4(positions[vid], 0.0, 1.0);
    }

    fragment float4 fragment_main() {
        return float4(1.0, 0.0, 0.0, 1.0);
    }
    """

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    private let pipeline: MTLRenderPipelineState
    private let queue: MTLCommandQueue

    init?(view: MTKView) {
        guard let device = view.device,
            let queue = device.makeCommandQueue(),
            let library = try? device.makeLibrary(source: metalSource, options: nil),
            let vertex = library.makeFunction(name: "vertex_main"),
            let fragment = library.makeFunction(name: "fragment_main")
        else { return nil }

        self.queue = queue

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vertex
        desc.fragmentFunction = fragment
        desc.colorAttachments[0].pixelFormat = view.colorPixelFormat

        guard let p = try? device.makeRenderPipelineState(descriptor: desc)
        else { return nil }
        self.pipeline = p
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
            let rpd = view.currentRenderPassDescriptor,
            let cmd = queue.makeCommandBuffer(),
            let enc = cmd.makeRenderCommandEncoder(descriptor: rpd)
        else { return }

        enc.setRenderPipelineState(pipeline)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        cmd.present(drawable)
        cmd.commit()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
}
