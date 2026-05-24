import Foundation

public enum NAXShaderLibrary {
    public static var metallibURL: URL {
        guard let url = Bundle.module.url(forResource: "nax_shader", withExtension: "metallib") else {
            fatalError("nax_shader.metallib not found in NAXShaders bundle")
        }
        return url
    }
}
