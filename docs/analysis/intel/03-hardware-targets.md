# 03 — Hardware Targets and Architectures

## 3.1 The Intel Xe family landscape (2026)

| Generation | Product line | Example SKUs | VRAM | Status | Vector-search fit |
|---|---|---|---|---|---|
| **Xe-LP** (Gen12) | UHD 7xx, Iris Xe | integrated (laptop) | shared, typically 4–8 GB | Production | 🔴 Not a target — low VRAM, low bandwidth |
| **Xe-HPG** (Alchemist) | Arc A380/A580/A750/A770 | consumer discrete | 6–16 GB GDDR6 | Legacy, still sold | 🟡 Possible for dev/CI; low share |
| **Xe-HPC** (Ponte Vecchio) | Data Center GPU Max 1100/1550 | HBM2e | up to 128 GB | **EOL Jan 2026** | 🔴 Do not invest |
| **Xe2** (Battlemage) | Arc B570 / B580 / (B770 pending) | consumer discrete | 10–12 GB GDDR6 | Production (late 2024 →) | 🟢 Primary consumer target |
| **Xe2** (Arc Pro) | Arc Pro B60 / B65 / B70 | workstation | 20–32 GB GDDR6 | Production (March 2026) | 🟢 **Primary prosumer target** (32 GB, XMX engines) |
| **Xe-LPG / Xe2** integrated | Meteor Lake / Lunar Lake / Arrow Lake | laptop integrated | shared | Production | 🔴 Not a target |
| **Xe3P** (Crescent Island) | Data Center GPU (successor to Max) | up to 160 GB LPDDR5X | Samples Q3 2026, GA TBD | Pre-launch | 🟢 Future HPC target (watch list) |

## 3.2 Tiering recommendation

- **Tier 1 (must work at launch):**
  - Arc Pro B70 (32 GB) — the strongest VRAM-for-dollar Intel option for vector search in 2026.
  - Arc B580 (12 GB) — most common consumer Battlemage card; needed for community adoption.
- **Tier 2 (validated but less exercised):**
  - Arc A770 (16 GB) — Alchemist, aging, but still available.
  - Arc Pro B60 / B65 (20 GB) — prosumer segment.
- **Tier 3 (compiled but not gated in CI):**
  - Arc A380/A580 — low VRAM, likely starved for workloads we care about.
  - Integrated Xe (Iris Xe, Xe-LPG) — available to everyone with a modern Intel laptop, useful for smoke tests but not a production target.
- **Watch list (revisit later):**
  - Crescent Island — enormous memory capacity makes it interesting for large HNSW graphs; re-evaluate once silicon is in customers' hands.

## 3.3 Architecture details that influence kernel design

- **SIMD width:** Intel Xe GPUs execute SIMD-8, SIMD-16, or SIMD-32 depending on the kernel. The Intel Graphics Compiler (IGC) auto-selects. Avoid hard-coding subgroup sizes.
- **EU / Xe-core organization:** Alchemist uses 8 EUs per Xe-core; Battlemage uses 16 XVEs (Xe Vector Engines) per Xe-core. Occupancy tuning differs between the two families — let the compiler pick, profile with `oneTrace` or Vulkan's validation layers.
- **XMX engines** (Battlemage, Arc Pro): matrix extensions similar to Tensor Cores / Matrix Cores. Accessible via DPC++ `joint_matrix` or Vulkan's `VK_KHR_cooperative_matrix` extension. **Not required in v1.**
- **Local memory (SLM):** 64 KB per workgroup on Xe-HPG/Xe2. Comparable to CUDA shared memory / ROCm LDS. Reductions fit the same pattern as Metal threadgroup memory.
- **Subgroup extensions:** `VK_KHR_shader_subgroup_*` is well supported on Intel; use subgroup reduction intrinsics for dot-product-style loops.

## 3.4 Operating system support

| OS | Driver | Status |
|---|---|---|
| Windows 10/11 x64 | Intel Arc Graphics Driver | ✅ Stable, monthly updates |
| Linux x64 | Mesa ANV (Vulkan) + Intel Compute Runtime (NEO) for L0/OpenCL | ✅ Stable on Ubuntu 22.04+, Fedora 38+ |
| Linux x64 — kernel | `i915` (legacy) or `xe` driver (new, from kernel 6.8) | 🟡 Kernel module choice affects feature availability |
| macOS | — | ❌ No Intel discrete GPU driver from Apple since 2022 |

The `xe` kernel driver is a hard requirement for Xe-HPC (Ponte Vecchio) and is the future for Battlemage on Linux. This means a Linux CI runner must use a modern distribution (Ubuntu 24.04 LTS or newer) to exercise the full backend.

## 3.5 Why integrated Xe is not a target

Iris Xe integrated graphics are interesting for **smoke testing** (any developer laptop from 2021 onward has it), but they are not serious vector-search hardware:

- Shared memory with the CPU — no independent VRAM pool.
- Memory bandwidth capped by LPDDR4/LPDDR5 (~60–120 GB/s), 5–10× slower than discrete.
- Thermal/power throttling in laptops makes benchmarks noisy.

The project should **allow** integrated Xe to work (same Vulkan path), but **not advertise it** as a production target in [README.md](../../../README.md) or [docs/PERFORMANCE.md](../../../docs/PERFORMANCE.md).
