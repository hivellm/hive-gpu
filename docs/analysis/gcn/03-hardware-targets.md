# 03 — Hardware Targets and Architectures

The proposal defines support starting from `gfx900+`. Quick map:

| gfx | Family | Example GPUs | Wavefront | Typical use |
|---|---|---|---|---|
| gfx900 | Vega 10 | MI25, Radeon Pro WX 9100 | 64 | Legacy datacenter |
| gfx906 | Vega 20 | MI50, MI60, Radeon VII | 64 | Legacy datacenter |
| gfx908 | CDNA | MI100 | 64 | HPC |
| gfx90a | CDNA2 | MI210, MI250, MI250X | 64 | HPC / AI |
| gfx940 / gfx942 | CDNA3 | MI300A, MI300X | 64 | AI training |
| gfx1030 | RDNA2 | RX 6800/6900, Radeon Pro W6800 | 32 | Consumer / pro |
| gfx1100 | RDNA3 | RX 7900 XTX, Radeon Pro W7900 | 32 (dual-issue) | Consumer / pro |

## Practical consequences

1. **Wavefront size is not constant.** Kernels must handle `warpSize = 32` OR `64` at runtime. Avoid hard-coding `__shfl_xor(..., 32)` in the CUDA style. Use `warpSize` or `__AMDGCN_WAVEFRONT_SIZE`.
2. **LDS size varies by architecture:** 64 KB/CU on Vega and CDNA; 64 KB (split across workgroups) on RDNA. Design reductions to fit the smaller figure.
3. **Register pressure differs:** CDNA favors wider occupancy with more registers; RDNA is tighter. Let `hipcc` pick occupancy unless profiling says otherwise.
4. **Dual-issue VOPD on RDNA3:** `hipcc` generally captures the opportunity; verify with `rocprof` before hand-tuning.
5. **Matrix cores:** CDNA (MFMA) and RDNA3 (WMMA) expose tensor-like units. v1 does **not** target them — `rocblas` abstracts the good path for BLAS; custom kernels stay scalar.

## Tiering recommendation for delivery

- **Tier 1 (must work at launch):** gfx90a (MI200 — the common cloud AMD instance), gfx1030 (RX 6000 — most common dev hardware).
- **Tier 2 (validated but less exercised):** gfx908, gfx1100, gfx940, gfx942.
- **Tier 3 (compiled but untested):** gfx900, gfx906. Keep the `--offload-arch` entries for source compatibility but do not gate CI on them.

## Platform support

| OS | ROCm availability | Recommended stance |
|---|---|---|
| Linux (Ubuntu 22.04 / RHEL 9) | ✅ Official | Primary target |
| Windows | 🟡 HIP SDK since 2023, spottier driver coverage | Mark as experimental at v1 |
| macOS | ❌ Not available (AMD removed macOS drivers) | Explicitly unsupported |

The `#[cfg(...)]` gating should follow:

```rust
#[cfg(all(feature = "rocm", any(target_os = "linux", target_os = "windows")))]
pub mod rocm;
```

## Why wavefront variability matters for this project

The HNSW distance reduction in [src/shaders/metal_hnsw.metal](../../../src/shaders/metal_hnsw.metal) assumes SIMD groups that match Apple GPU families. When porting to HIP, the reduction must be written once and parameterized by `warpSize`:

```cpp
const unsigned wave = warpSize;  // 32 on RDNA, 64 on CDNA
for (unsigned offset = wave / 2; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
}
```

Testing on at least one RDNA (wave=32) *and* one CDNA (wave=64) card before merging is non-negotiable.
