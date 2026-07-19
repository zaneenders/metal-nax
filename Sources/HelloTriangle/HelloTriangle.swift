import AppKit
import Metal
import MetalKit

@main
struct HelloTriangle {
    public static func main() {
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
    }
}
