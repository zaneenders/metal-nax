# Hello Triangle

Minimal Metal graphics — one file, no build plugins.

## Run

```bash
swift run -c release
```

A window opens with a red triangle on a dark blue background.

## Structure

```
Package.swift              # single executable, links Metal + MetalKit + AppKit
Sources/HelloTriangle/main.swift  # shaders, renderer, app — all in one file
```

The Metal shader source is embedded as a Swift string and compiled at runtime via
`MTLDevice.makeLibrary(source:)`. No `.metal` files, no build plugins.
