// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "NAXDemo",
    platforms: [
        .macOS(.v26),
    ],
    products: [
        .executable(name: "nax-demo", targets: ["NAXDemo"]),
    ],
    targets: [
        .executableTarget(
            name: "NAXDemo",
            dependencies: ["NAXShaders"],
            linkerSettings: [
                .unsafeFlags(
                    ["-framework", "Metal", "-framework", "MetalPerformanceShaders"],
                    .when(platforms: [.macOS])
                ),
            ]
        ),
        .target(
            name: "NAXShaders",
            exclude: ["Shaders"],
            plugins: [
                .plugin(name: "NAXMetalPlugin"),
            ]
        ),
        .plugin(
            name: "NAXMetalPlugin",
            capability: .buildTool()
        ),
    ]
)
