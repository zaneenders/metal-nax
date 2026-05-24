import Foundation
import PackagePlugin

@main
struct NAXMetalPlugin: BuildToolPlugin {
    func createBuildCommands(
        context: PluginContext,
        target: Target
    ) async throws -> [Command] {
        let metalFiles = try metalSourceFiles(for: target)
        guard !metalFiles.isEmpty else { return [] }

        let workDirectory = context.pluginWorkDirectoryURL
        let xcrun = URL(fileURLWithPath: "/usr/bin/xcrun")

        var airOutputs: [URL] = []
        var compileCommands: [Command] = []

        for metalFile in metalFiles {
            let stem = metalFile.deletingPathExtension().lastPathComponent
            let airOutput = workDirectory.appending(path: "\(stem).air")
            airOutputs.append(airOutput)

            compileCommands.append(
                .buildCommand(
                    displayName: "Compile Metal shader \(metalFile.lastPathComponent)",
                    executable: xcrun,
                    arguments: [
                        "-sdk", "macosx",
                        "metal", "-c",
                        metalFile.path,
                        "-o", airOutput.path,
                    ],
                    inputFiles: [metalFile],
                    outputFiles: [airOutput]
                )
            )
        }

        let metallibOutput = workDirectory.appending(path: "nax_shader.metallib")
        let linkCommand = Command.buildCommand(
            displayName: "Link nax_shader.metallib",
            executable: xcrun,
            arguments: [
                "-sdk", "macosx",
                "metallib",
            ] + airOutputs.map(\.path) + [
                "-o", metallibOutput.path,
            ],
            inputFiles: airOutputs,
            outputFiles: [metallibOutput]
        )

        return compileCommands + [linkCommand]
    }
}

private func metalSourceFiles(for target: Target) throws -> [URL] {
    let shadersDirectory = target.directoryURL.appending(path: "Shaders")
    if FileManager.default.fileExists(atPath: shadersDirectory.path) {
        return try FileManager.default.contentsOfDirectory(atPath: shadersDirectory.path)
            .filter { $0.hasSuffix(".metal") }
            .sorted()
            .map { shadersDirectory.appending(path: $0) }
    }

    guard let sourceModule = target.sourceModule else { return [] }
    return sourceModule.sourceFiles(withSuffix: "metal").map(\.url)
}
