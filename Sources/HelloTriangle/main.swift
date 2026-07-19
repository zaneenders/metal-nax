import AppKit
import Metal
import MetalKit

// MARK: - Shader source (compiled at runtime — no build plugin needed)

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

// MARK: - Renderer

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

// MARK: - App

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let view = MTKView(frame: NSRect(x: 0, y: 0, width: 800, height: 600))
view.device = MTLCreateSystemDefaultDevice()
view.clearColor = MTLClearColor(red: 0.1, green: 0.1, blue: 0.2, alpha: 1.0)

guard let renderer = Renderer(view: view) else {
    fatalError("Metal requires Apple Silicon or supported GPU.")
}
view.delegate = renderer

let window = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
    styleMask: [.titled, .closable, .resizable],
    backing: .buffered,
    defer: false
)
window.title = "Hello Triangle"
window.contentView = view
window.center()
window.makeKeyAndOrderFront(nil)

app.activate(ignoringOtherApps: true)
app.run()
