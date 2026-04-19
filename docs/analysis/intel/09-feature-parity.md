# 09 — Feature Parity Matrix

Expected feature coverage once the Intel backend ships at v0.4.0.

| Feature | Metal (today) | CUDA (planned) | ROCm (planned) | Intel (planned) |
|---|---|---|---|---|
| Context creation | ✅ | ✅ | ✅ | ✅ (Vulkan instance + device) |
| Real device info | ✅ | ✅ | ✅ | ✅ (VkPhysicalDeviceProperties) |
| VRAM-only storage | ✅ | ✅ | ✅ | ✅ (device-local VkBuffer) |
| Add single / batch vectors | ✅ | ✅ | ✅ | ✅ |
| Brute-force L2 search | 🟡 (mock) | ✅ | ✅ | ✅ (hand-written GLSL) |
| Brute-force Cosine / Dot | 🟡 | ✅ (cuBLAS) | ✅ (rocBLAS) | ✅ (hand-written, no BLAS) |
| HNSW construction / search | 🟡 (partial) | ❌ v1 | ❌ v1 | ❌ v1 |
| Dynamic buffer expansion | ✅ | ✅ | ✅ | ✅ |
| Cross-backend consistency | N/A | 🎯 tested | 🎯 tested | 🎯 tested |
| XMX / Tensor / Matrix cores | ❌ | ❌ v1 | ❌ v1 | ❌ v1 |

## Expected performance envelope

Rough expectation relative to a well-tuned CUDA baseline on equivalent silicon price:

| Backend | Relative perf | Rationale |
|---|---|---|
| CUDA on RTX 4090 | 1.0× (reference) | Native, cuBLAS |
| ROCm on MI210 | 0.9–1.0× | Native, rocBLAS |
| Metal on M3 Max | 0.7–0.9× | Native, but smaller memory bandwidth |
| Intel on Arc Pro B70 | 0.5–0.7× | Vulkan overhead + no BLAS; XMX unused in v1 |
| Intel on Arc B580 | 0.3–0.5× | Consumer silicon, smaller memory bus |
| Vulkan fallback on NVIDIA/AMD | 0.5–0.7× of native | Same code, vendor path not optimal |

**These are guesses pre-benchmark.** Replace with real numbers at the end of Phase 5.

## Numerical tolerance envelope (unchanged from ROCm)

| Metric | Tolerance per element | Top-K order |
|---|---|---|
| L2 | 1e-4 absolute | Top-10 set equality (ties may swap) |
| Cosine | 1e-5 absolute after normalization | Same |
| DotProduct | 1e-4 relative | Same |

Specific known sources of divergence to document:

- FMA fusion differences between Intel's IGC compiler and NVIDIA's/AMD's.
- Denormal-flushing defaults (Intel IGC defaults to flush-to-zero).
- `inversesqrt()` approximation quality varies between Vulkan drivers.

## Operations intentionally not accelerated in v1

Same posture as ROCm — everything below stays on CPU until a v2 scope is funded:

| Operation | v1 behavior | v2 target |
|---|---|---|
| Top-K on GPU | CPU-side sort after readback | Vulkan compute radix sort |
| Remove vectors | Soft-delete via index mask | Periodic compaction pass |
| Multi-GPU | Single device per context | Device selection API |
| Quantization (PQ / SQ) | Not present | Phase 4 in the main roadmap |
| Filtered search | Not present | Phase 4 |
| XMX matrix acceleration | Not used | `VK_KHR_cooperative_matrix` path |

## What the Intel backend uniquely offers

These are not features per se, but reasons to enable the Intel backend even on non-Intel hardware:

- **Universal fallback:** `HIVE_GPU_VULKAN_UNIVERSAL=1` lets the same backend run on NVIDIA/AMD when their native toolchains are missing (e.g. Docker containers without CUDA drivers installed).
- **Validation-layer debugging:** Vulkan validation layers catch synchronization bugs that CUDA/ROCm equivalents miss. Useful during development even if production uses a native backend.
- **Portability of kernels:** a SPIR-V kernel is vendor-neutral. If a future vendor ships a Vulkan driver (Imagination, Samsung, Qualcomm), the Intel backend runs on it for free.

Those side benefits are real but should not be the primary justification for the work. The primary justification is Intel Arc Pro customers.
