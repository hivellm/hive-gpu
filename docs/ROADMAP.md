# hive-gpu — Roadmap

**Last updated:** 2026-04-19

## Current status

- **Version:** 0.2.0 (released 2026-04-19)
- **Backends shipping:**
  - **Metal** on macOS / Apple Silicon — stable since 0.1.x
  - **CUDA** on Linux / Windows / NVIDIA (Volta+ / sm_70+) — new in 0.2.0,
    validated end-to-end on RTX 4090
- **Backends designed but not implemented:**
  - **ROCm** for AMD — analysis at [`docs/analysis/gcn/`](analysis/gcn/),
    task at [`.rulebook/tasks/phase3b_add-rocm-backend/`](../.rulebook/tasks/phase3b_add-rocm-backend/)
  - **Intel** (Arc / Battlemage) via Vulkan Compute — analysis at
    [`docs/analysis/intel/`](analysis/intel/), task at
    [`.rulebook/tasks/phase3c_add-intel-backend/`](../.rulebook/tasks/phase3c_add-intel-backend/)
- **Production readiness:** 0.2.x is **beta**. API surface stable;
  0.3-series focuses on HNSW, quantization, and additional vendors.

## Market coverage

| Vendor  | Backend | Status  | Share of ML/AI GPU market |
|---------|---------|---------|--------------------------:|
| Apple   | Metal   | ✅ shipping | ~5 % (Apple Silicon)  |
| NVIDIA  | CUDA    | ✅ shipping | ~70 %                 |
| AMD     | ROCm    | 📝 designed | ~15 %                 |
| Intel   | Vulkan  | 📝 designed | <1 % (today)          |
| Any     | CPU     | ✅ fallback | universal             |

Metal + CUDA = **~75 % coverage today**; adding ROCm reaches ~90 %.

---

## ✅ Phase 1 — Foundation (v0.1.x)

Completed across the 0.1.x series (2024–2025).

- [x] Core traits and types (`GpuBackend`, `GpuContext`, `GpuVectorStorage`)
- [x] Metal Native backend (Apple Silicon) via `objc2-metal`
- [x] VRAM-only vector storage with adaptive buffer growth
- [x] Distance metrics: Cosine, Euclidean, DotProduct
- [x] Error handling via `thiserror`
- [x] Migration from discontinued `metal-rs` to `objc2-metal` (0.1.8)
- [x] 72-test integration suite on Metal (0.1.9)
- [x] Device Info API: `GpuDeviceInfo` struct, `device_info()` on `GpuContext`,
      helpers (`vram_usage_percent`, `has_available_vram`) — 0.1.7

**Open items carried forward into Phase 3:**

- [ ] GPU-accelerated HNSW construction (Metal has partial, CUDA is brute-force only)
- [ ] Product / Scalar quantization

---

## ✅ Phase 2 — CUDA Backend (v0.2.0) — SHIPPED

Released 2026-04-19.

- [x] `cudarc 0.13` integration with `dynamic-linking` (no CUDA Toolkit
      required at build time on end-user machines)
- [x] `CudaContext` with live driver queries: compute capability,
      total/free VRAM, driver version, PCI bus id
- [x] `CudaVectorStorage` with batched `htod_copy`, adaptive
      device-to-device reallocation, soft-delete, host-cached norms
- [x] GPU search via cuBLAS SGEMV (DotProduct / Cosine / Euclidean)
- [x] 17 integration tests (`cuda_smoke`, `cuda_device_info`,
      `cuda_vector_ops`) passing on RTX 4090
- [x] Criterion benchmark `benches/cuda_ops.rs` comparing GPU vs CPU
- [x] GitHub Actions workflow against `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- [x] `HiveGpuError::{CudaError, CublasError}` variants
- [x] `#![allow(warnings)]` removed project-wide; clippy `-D warnings`
      enforced on all feature combinations

**Validated performance on RTX 4090 (DotProduct, 128-dim, top-10):**

| N       | GPU    | CPU baseline | Speedup |
|--------:|-------:|-------------:|--------:|
|   1 000 | 124 µs | 63 µs        | 0.5×    |
|  10 000 | 287 µs | 690 µs       | 2.4×    |
| 100 000 | 4.0 ms | 13.0 ms      | 3.25×   |

See [`benchmarks/PERFORMANCE.md`](benchmarks/PERFORMANCE.md) for full numbers.

---

## 📝 Phase 3 — ROCm Backend (v0.3.0) — PLANNED

**Priority:** ⚡ High · **Est. effort:** 16–21 dev-days after CUDA
stabilises · **Task:** [`phase3b_add-rocm-backend`](../.rulebook/tasks/phase3b_add-rocm-backend/)
· **Analysis:** [`docs/analysis/gcn/`](analysis/gcn/)

### Deliverables

- [ ] `src/rocm/` module behind a `rocm` Cargo feature (Linux / Windows
      target-gated)
- [ ] HIP bindings via `bindgen` against `$ROCM_PATH/include/hip/` plus
      `rocblas-sys` for BLAS routines
- [ ] `RocmContext` with HIP stream + rocBLAS handle; device info from
      `hipGetDeviceProperties` exposing the real gfx architecture string
      (gfx900 through gfx1100+)
- [ ] `RocmVectorStorage` mirroring the CUDA shape: `hipMalloc`,
      `hipMemcpyAsync`, D2D reallocation on growth
- [ ] `src/rocm/kernels.hip` for L2 distance (Cosine and DotProduct route
      through rocBLAS SGEMV), compiled multi-arch via `hipcc
      --offload-arch=gfx900,gfx906,gfx908,gfx90a,gfx1030,gfx1100`
- [ ] Runtime wavefront-size awareness (32 on RDNA, 64 on CDNA)
- [ ] Cross-backend consistency tests Metal × CUDA × ROCm within 1e-4
- [ ] CI job using `rocm/dev-ubuntu-22.04:6.0` container for build
      verification
- [ ] Examples and `docs/guides/ROCM_SETUP.md`

### Dependencies

- Hardware access: gfx1030 (RX 6000 / RDNA2) minimum for dev, ideally also
  gfx90a (MI200 / CDNA2) for wavefront-64 validation before release.
- Stable CUDA backend to mirror its storage / search shape.

---

## 📝 Phase 4 — Intel Backend (v0.3.x) — PLANNED

**Priority:** 📦 Low · **Est. effort:** 22–28 dev-days · **Task:**
[`phase3c_add-intel-backend`](../.rulebook/tasks/phase3c_add-intel-backend/)
· **Analysis:** [`docs/analysis/intel/`](analysis/intel/)

### Deliverables

- [ ] `src/intel/` module behind an `intel` Cargo feature
- [ ] Vulkan Compute via `ash 0.37+`, physical-device filter by
      `vendorID == 0x8086`
- [ ] GLSL compute shaders (or WGSL via Naga) compiled to SPIR-V in
      `build.rs` with `shaderc`, embedded via `include_bytes!`
- [ ] `IntelContext` / `IntelVectorStorage` following the Metal / CUDA /
      ROCm pattern
- [ ] Hand-written SGEMV kernel (no BLAS path available from Rust on
      Intel)
- [ ] Universal Vulkan fallback mode (`HIVE_GPU_VULKAN_UNIVERSAL=1`) that
      also runs on NVIDIA / AMD drivers for Docker images without vendor
      SDKs
- [ ] Cross-backend consistency including Intel within 1e-4
- [ ] CI job using `lavapipe` for software-Vulkan build verification

### Hardware targets (tiered)

- **Tier 1:** Arc Pro B70 (32 GB GDDR6) — primary workstation target.
- **Tier 1:** Arc B580 (12 GB GDDR6) — primary consumer Battlemage.
- **Tier 2:** Arc A770 / B60 / B65 — validated, lower priority.
- **Future:** Crescent Island (Xe3P, 160 GB LPDDR5X) — sampling Q3 2026.

Performance expectation: 40–60 % of native CUDA/ROCm on equivalent silicon
price. Documented honestly — this is not meant to replace a vendor-native
path on that vendor's hardware.

### Dependencies

- Hardware access: at least Arc B580 on a dev workstation; ideally a
  self-hosted CI runner with an Arc Pro B70.
- ROCm backend shipped first so the `build.rs` shader-compile pattern is
  established.
- Commitment to maintaining a fourth backend post-release.

---

## Phase 5 — Advanced Features (v0.4.0) — PLANNED ⚡

**Priority:** ⚡ Medium · **Est. effort:** 3–4 months

### Vector quantization

- [ ] Product Quantization (PQ)
  - [ ] GPU-accelerated codebook training
  - [ ] Fast PQ encoding on GPU
  - [ ] Asymmetric distance computation
- [ ] Scalar Quantization (SQ8, SQ6, SQ4)
  - [ ] GPU-accelerated quantization
- [ ] Binary Quantization
  - [ ] Hamming distance on GPU

### HNSW on the GPU

- [ ] GPU-accelerated HNSW construction on CUDA
- [ ] GPU-accelerated HNSW search on CUDA (port Metal's `metal_hnsw.metal`)
- [ ] Matching implementation on ROCm
- [ ] Dynamic graph updates and batch insertion
- [ ] Memory-efficient graph storage
- [ ] Level-wise parallel construction

### GPU-side top-K

- [ ] CUDA: `cub::DeviceRadixSort` or hand-rolled radix-select
- [ ] ROCm: equivalent via rocPRIM
- [ ] Metal / Intel: parallel radix select compute kernels
- [ ] Remove the CPU-side sort fallback from the hot path

### Extended distance metrics

- [ ] Manhattan distance (L1)
- [ ] Chebyshev distance (L∞)
- [ ] Custom user-supplied distance functions
- [ ] Mixed-precision distance computation (f16 / bf16)

### Memory management

- [ ] Adaptive buffer sizing
- [ ] Memory pressure handling (spill to CPU)
- [ ] Transparent compression for inactive data
- [ ] VRAM defragmentation / compaction pass

### Observability

- [ ] Real-time performance metrics via `tracing` spans
- [ ] Throughput / latency histograms
- [ ] VRAM usage tracking endpoint
- [ ] Optional Tracy / Instruments profiling integration

---

## Phase 6 — Production Readiness (v1.0.0) — PLANNED 📦

**Priority:** 📦 Triggered when v0.4 has been stable for at least one minor
release · **Est. effort:** 2–3 months

### Reliability

- [ ] Comprehensive error recovery paths
- [ ] Graceful degradation to CPU fallback under VRAM pressure
- [ ] Health check endpoint for server integrations
- [ ] Automatic retry logic for transient driver errors

### Scalability

- [ ] Multi-GPU support: per-context device selection
- [ ] Intra-process load balancing across GPUs
- [ ] Sharding strategies for > single-GPU VRAM datasets
- [ ] Replication hooks

### Integrations

- [ ] First-class `hive-vectorizer` integration
- [ ] Qdrant / Milvus / Weaviate plugins
- [ ] LangChain vector-store adapter

### Observability (hardening)

- [ ] OpenTelemetry exporter
- [ ] Prometheus metrics
- [ ] Structured logging
- [ ] Distributed tracing

### Quality gates for 1.0

- [ ] ≥ 95 % code coverage
- [ ] Long-running stability tests (24 h+)
- [ ] Chaos-engineering style fault injection
- [ ] Stress suite: millions of vectors, sustained QPS
- [ ] Memory-leak audit (valgrind / compute-sanitizer) clean

---

## Future vision (v2.x+)

- Distributed vector search with consensus-based replication
- Geographic distribution / cross-datacenter search
- GPU-accelerated embedding generation (host the encoder on-device)
- Hybrid search (dense + sparse) on GPU
- Filtered search with GPU-side predicate evaluation
- Next-generation hardware: Apple Neural Engine, AMD Instinct MI400,
  Intel Crescent Island, TPUs

---

## Timeline

```
2024 Q4 – 2025 Q4     ✅ Phase 1 foundation (0.1.x)
                         Metal backend, device info API, 72-test suite

2026 Q2 (Apr)         ✅ Phase 2 CUDA shipped (0.2.0)
                         cuBLAS SGEMV, 17 CUDA tests, RTX 4090 validation

2026 Q2 – Q3          📝 Phase 3 ROCm (0.3.0)
                         HIP + rocBLAS, gfx900+ coverage

2026 Q3 – Q4          📝 Phase 4 Intel (0.3.x)
                         Vulkan Compute + SPIR-V, Arc Pro target

2026 Q4 – 2027 Q1     Phase 5 advanced features (0.4.x)
                         Quantization, GPU HNSW, GPU top-K

2027 Q2+              Phase 6 production readiness (1.0.0)
```

Dates are estimates anchored to engineering capacity; no hard external
commitments.

---

## Versioning policy

Semantic versioning ([semver.org](https://semver.org/)):

- **MAJOR** — breaking API changes (e.g. 1.0.0 → 2.0.0)
- **MINOR** — new features, backwards compatible (0.2.0 → 0.3.0)
- **PATCH** — bug fixes (0.2.0 → 0.2.1)

| Version | Status   | Support window     | Notes                               |
|---------|----------|--------------------|-------------------------------------|
| 0.1.x   | Archived | Ended with 0.2.0   | Metal-only; superseded              |
| 0.2.x   | Active   | Until 0.4.0 ships  | Metal + CUDA; public API stable     |
| 0.3.x   | Planned  | 9 months post-1.0  | Adds ROCm and Intel                 |
| 0.4.x   | Planned  | 12 months post-1.0 | Advanced features                   |
| 1.0.x   | Planned  | LTS                | Production-ready                    |

---

## Priorities and decision rules

Features are prioritised by:

1. **Impact** — how many users benefit, how large the market slice?
2. **Dependencies** — blocked by other features?
3. **Maintenance cost** — can we keep it green after release?
4. **Strategic fit** — aligned with the "universal vector search" goal?
5. **Community demand** — number of requests and votes on GitHub.

A feature lands only if all five signals are positive. Hardware backends
additionally require guaranteed access to the target silicon for both
development and ongoing CI validation — shipping a backend we cannot test
regressed is worse than not shipping it.

---

## Risks and mitigations

### Technical

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Vendor driver ABI drift (cudarc / HIP) | Medium | Medium | Pin minor versions, test matrix across CUDA 12.0 – 13.x |
| Numerical divergence across backends | Medium | Medium | 1e-4 cross-backend consistency tests per release |
| VRAM fragmentation after many expansions | Low | Medium | Compaction pass planned in 0.4 |
| CI runners without GPU hide regressions | High | Low | Every GPU test is a no-op when no driver; self-hosted runner for nightly |

### Organisational

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| No AMD hardware for ROCm dev/CI | High | High | Secure cloud access (MI300 on GCP / Azure) before Phase 3 starts |
| No Intel Arc hardware for Phase 4 | High | High | Workstation provisioning gate before any Intel code lands |
| Maintainer bandwidth for four backends | Medium | High | Feature-gate everything; document abort triggers per analysis |

---

## Contributing

We welcome community input on the roadmap:

1. **Request features:** open a GitHub issue tagged `[FEATURE REQUEST]`.
2. **Propose changes:** submit a PR to this document.
3. **Vote:** react with 👍 on issues you want prioritised.
4. **Discuss:** join GitHub Discussions.

For implementation work, pick up one of the phased tasks under
[`.rulebook/tasks/`](../.rulebook/tasks/) and the corresponding analysis
under [`docs/analysis/`](analysis/).
