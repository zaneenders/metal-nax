# Metal 4 NAX (Neural Acceleration eXtension) Example

A minimal **Metal 4** compute shader that leverages **NAX**, the dedicated Neural Accelerator inside the Apple M5 GPU.

## What is NAX?

NAX is a hardware feature built into the M5 GPU — a dedicated **Neural Accelerator** on every GPU core that speeds up:

- **bfloat16** (BF16) matrix multiply–accumulate (GEMM)
- Quantized INT8 / INT4 tensor ops
- Fused activation functions (ReLU, GELU, etc.)

You don't write raw NAX assembly. You write high-level **Metal Performance Primitives (MPP)** tensor ops (`matmul2d`, `convolution2d`, etc.) using the `bfloat` type, and the Metal compiler/driver automatically maps them to NAX instructions on M5 hardware.

## Project Structure

```
metal-nax/
├── build/                  # Build artifacts (ignored by git)
│   ├── nax_demo            # Runnable binary
│   ├── nax_shader.air      # Compiled shader IR
│   └── nax_shader.metallib # Compiled shader library
├── nax_shader.metal        # Metal 4 compute kernels
├── NAXHost.swift           # Swift host code (pipeline, dispatch)
├── .gitignore              # Ignores build/ and macOS noise
└── README.md               # This file
```

## Quick Start

### 1. Build

Make sure Xcode is your active developer directory:

```bash
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

Then compile everything:

```bash
mkdir -p build

# Compile the Metal shader
xcrun -sdk macosx metal -c nax_shader.metal -o build/nax_shader.air
xcrun -sdk macosx metallib build/nax_shader.air -o build/nax_shader.metallib

# Compile the Swift host in release mode
swiftc -O -framework Metal -framework MetalPerformanceShaders NAXHost.swift -o build/nax_demo
```

### 2. Run

```bash
./build/nax_demo
```

Output:
```
Device: Apple M5
Supports NAX (Metal 4): true
NAX kernel completed: 128x64 output computed.
```

### 3. Rebuild

After changing shader or host code, just rerun the build steps above. The binary is self-contained — it loads `build/nax_shader.metallib` at runtime.

---

## How It Works

### Shader: `naxFusedMatmulBiasRelu`

The kernel computes **C = ReLU(A × B + bias)** in one shot:

```metal
kernel void naxFusedMatmulBiasRelu(
    tensor<device bfloat, dextents<int32_t, 2>> A,     // M × K
    tensor<device bfloat, dextents<int32_t, 2>> B,     // K × N
    tensor<device bfloat, dextents<int32_t, 2>> bias,  // 1 × N
    tensor<device bfloat, dextents<int32_t, 2>> C,     // M × N output
    ...
)
```

**Key techniques:**

| Technique | What it does |
|-----------|--------------|
| **`bfloat`** | 16-bit float with full `float` dynamic range. The only type that triggers the NAX fast path. |
| **`matmul2d`** | MPP high-level GEMM. Tiled into 64×32 output tiles with 4 cooperating SIMD groups. |
| **`relaxed_precision=true`** | Tells the driver it's okay to use approximate-but-fast NAX instructions. |
| **Cooperative tensors** | Keeps the matmul result in registers so bias+ReLU are fused with **zero extra memory traffic**. |

### Dispatch Grid

The host dispatches exactly what the shader expects:

```swift
let threadsPerThreadgroup = MTLSize(
    width: simdgroupWidth * 4,  // 4 SIMD groups = 128 threads
    height: 1,
    depth: 1
)

let threadgroups = MTLSize(
    width:  (N + 31) / 32,   // 32 columns per tile
    height: (M + 63) / 64,   // 64 rows per tile
    depth:  1
)
```

> ⚠️ **Mismatching the SIMD-group count is undefined behavior.** The shader declares `execution_simdgroups<4>`; the host must match.

---

## Performance Tips

1. **Always use `bfloat`**, `half` and `float` won't hit the NAX path.
2. **Enable `relaxed_precision`** in `matmul2d_descriptor` to let the driver choose the fastest kernel.
3. **Fuse everything in registers** with cooperative tensors. Every round-trip to device memory costs bandwidth and latency.
4. **Tile sizes matter** — 64×32 with 4 SIMD groups is a good starting point for M5, but profile your actual shapes.
5. **Use `static_slice` for known bounds** — If K is fixed at compile time, replace `dynamic_extent` to skip runtime bounds checks.

