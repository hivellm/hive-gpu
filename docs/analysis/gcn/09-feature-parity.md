# 09 — Feature Parity Matrix

Expected feature coverage after the plan lands.

| Feature | Metal (today) | CUDA (planned) | ROCm (planned) |
|---|---|---|---|
| Context creation | ✅ | ✅ | ✅ |
| Real device info | ✅ | ✅ | ✅ (via `hipGetDeviceProperties`) |
| VRAM-only storage | ✅ | ✅ | ✅ |
| Add single / batch vectors | ✅ | ✅ | ✅ |
| Brute-force L2 search | 🟡 (mock in `search`) | ✅ | ✅ |
| Brute-force Cosine / Dot search | 🟡 | ✅ (cuBLAS) | ✅ (rocBLAS) |
| HNSW construction / search | 🟡 (partial) | ❌ v1 | ❌ v1 |
| Dynamic buffer expansion | ✅ | ✅ | ✅ |
| Cross-backend consistency | N/A | 🎯 tested | 🎯 tested |

## HNSW is deferred for both CUDA and ROCm

Porting [metal_hnsw.metal](../../../src/shaders/metal_hnsw.metal) is a separate project (~1–2 weeks per backend). The strategic call is to ship brute-force-only CUDA and ROCm first, because:

1. Brute-force already delivers the advertised 10–50× speedup vs CPU.
2. HNSW adds algorithmic complexity that amplifies debugging cost on three backends.
3. The consistency harness is easier to validate when all three backends run the same algorithm.

Once HNSW is stable on Metal (currently partial), it can be ported to the other backends in parallel by following the same kernel layout.

## Numerical tolerance envelope

Agreed thresholds for cross-backend equivalence tests, pending empirical calibration:

| Metric | Tolerance per element | Top-K order |
|---|---|---|
| L2 | 1e-4 absolute | Top-10 must be identical sets (order may swap for equal distances) |
| Cosine | 1e-5 absolute after normalization | Same |
| DotProduct | 1e-4 relative | Same |

Divergences beyond these are treated as bugs, not floating-point noise. Specific known causes to document:

- FMA fusion differences (Metal vs CUDA vs ROCm).
- Denormal flushing defaults per vendor.
- `rsqrt` approximations in Cosine normalization.

## Operations shipped but intentionally not optimized in v1

| Operation | v1 behavior | v2 target |
|---|---|---|
| Top-K on GPU | CPU-side sort after copy-back | `cub::DeviceRadixSort` (CUDA) / equivalent on ROCm |
| Remove vectors | Soft-delete via index mask | Periodic compaction pass |
| Multi-GPU | Single device per context | `Arc<Device>` with shard routing |
| Quantization (PQ / SQ) | Not present | Phase 4 in the main roadmap |
| Filtered search | Not present | Phase 4 |

This matrix should be mirrored in [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md) when ROCm ships so users can predict what is and is not accelerated.
