# DGX Spark GEMMs — CLAUDE.md

## Project Goal

Implement CUTLASS-based C++ kernels for the NVIDIA NVF4 Kernel Hackathon hosted by GPU Mode.
Competition: https://forums.developer.nvidia.com/t/join-us-for-the-blackwell-nvfp4-kernel-hackathon-with-nvidia-and-gpu-mode/350092
Reference kernels: https://github.com/gpu-mode/reference-kernels/tree/main/problems/nvidia

## Hardware Target

- **Device**: NVIDIA DGX Spark
- **Architecture**: SM121a (Blackwell GeForce, compute capability 12.1)
- **Benchmark clock**: 1.5 GHz
- **Architecture is hardcoded** in CMakeLists.txt — do not add SM version flags

## Build System

```bash
cmake -B build && cmake --build build
# Build single target:
cmake --build build --target <target_name>
```

Key CMake flags applied globally:
- `--expt-relaxed-constexpr` — required for blockwise scaling headers (87_* kernels)
- `-std=c++17`

## Data Formats

### NVFP4 — `float4_e2m1fn_x2`
- 4-bit float: 1 sign + 2 exponent + 1 mantissa (e2m1)
- Two FP4 values packed per byte
- Values: `{-1.5, -1, -0.5, 0, +0.5, +1, +1.5}`
- K-major storage: physical K dimension is `k // 2` bytes wide

### Scale Factors — `float8_e4m3fn`
- One FP8 scale per 16 consecutive FP4 elements along K: `sf_k = ceil_div(k, 16)`
- Logical shape: `[MN, K//16, L]`
- **Reordered/permuted layout** for MMA (what custom kernels receive):
  - Shape: `[32, 4, ceil_div(MN,128), 4, ceil_div(K//16,4), L]`
  - MMA atom: `atom_m = (32, 4)`, `atom_k = 4`
  - Mapping from logical `(i, j, b)` → reordered `(mm32, mm4, mm, kk4, kk, b)`:
    - `mm32 = i % 32`
    - `mm4  = (i % 128) // 32`
    - `mm   = i // 128`
    - `kk4  = j % 4`
    - `kk   = j // 4`
  - This matches the cuBLAS FP4 block-scaling-factors layout

## Competition Problems

All 4 problems target B200 (SM121a), ranked by geometric mean of benchmark timings.
Correctness tolerance: `rtol=1e-03, atol=1e-03`.
Speed-of-light baseline: `max(FP4 Tensor Core throughput, DRAM throughput)` at 1.5 GHz.

---

### Problem 1: `nvfp4_gemv` — Batched FP4 GEMV

**Operation**: Batched matrix-vector multiply with NVFP4 block scaling.

**Inputs** (7-tuple):
```
a             : [M, K//2, L]          float4_e2m1fn_x2   (K-major)
b             : [1, K//2, L]          float4_e2m1fn_x2   (K-major, N=1 vector; padded to 128 internally)
sfa           : [M, K//16, L]         float8_e4m3fn       (reference layout, on CPU)
sfb           : [1, K//16, L]         float8_e4m3fn       (reference layout, on CPU; N padded to 128)
sfa_permuted  : [32,4,ceil(M/128),4,ceil(K//16/4),L]  float8_e4m3fn  (MMA layout, on GPU)
sfb_permuted  : [32,4,ceil(128/128),4,ceil(K//16/4),L]  float8_e4m3fn  (MMA layout, on GPU)
c             : [M, 1, L]             float16             (output, in-place)
```

**Constraints**: M divisible by `mma_tiler_mn[0]`, K divisible by **64**.

**Benchmarks** (speed-of-light targets at 1.5 GHz):
| M    | K     | L | Time (μs) |
|------|-------|---|-----------|
| 7168 | 16384 | 1 | 8.622     |
| 4096 | 7168  | 8 | 17.275    |
| 7168 | 2048  | 4 | 4.317     |

**Note**: `91_fp4_gemv` in `sample_kernels/` is a **single-batch** GEMV reference — useful for the CUTLASS kernel structure, but the competition kernel must handle the batch dimension L.

---

### Problem 2: `nvfp4_gemm` — FP4 Block-Scaled GEMM

**Operation**: `C = A @ B^T` with NVFP4 block scaling, output in FP16.

**Inputs** (7-tuple):
```
a             : [M, K//2, L]          float4_e2m1fn_x2
b             : [N, K//2, L]          float4_e2m1fn_x2
sfa           : [M, K//16, L]         float8_e4m3fn       (reference layout)
sfb           : [N, K//16, L]         float8_e4m3fn       (reference layout)
sfa_permuted  : [32,4,ceil(M/128),4,ceil(K//16/4),L]  float8_e4m3fn
sfb_permuted  : [32,4,ceil(N/128),4,ceil(K//16/4),L]  float8_e4m3fn
c             : [M, N, L]             float16             (output, in-place)
```

**Constraints**: M divisible by `mma_tiler_mn[0]`, N divisible by `mma_tiler_mn[1]`, K divisible by **256**.

**Benchmarks** (speed-of-light targets at 1.5 GHz):
| M   | N    | K     | L | Time (μs) |
|-----|------|-------|---|-----------|
| 128 | 7168 | 16384 | 1 | 8.994     |
| 128 | 4096 | 7168  | 1 | 2.354     |
| 128 | 7168 | 2048  | 1 | 1.333     |

**Relevant sample kernels**: `79a`, `79b`, `79c` in `sample_kernels/`.

---

### Problem 3: `nvfp4_dual_gemm` — FP4 Dual GEMM with SiLU

**Operation**: `C = silu(A @ B1^T) * (A @ B2^T)`, fused with SiLU activation (SwiGLU-style MLP gate).

**Inputs** (10-tuple):
```
a              : [M, K//2, L]          float4_e2m1fn_x2
b1             : [N, K//2, L]          float4_e2m1fn_x2
b2             : [N, K//2, L]          float4_e2m1fn_x2
sfa            : [M, K//16, L]         float8_e4m3fn       (reference layout)
sfb1           : [N, K//16, L]         float8_e4m3fn       (reference layout)
sfb2           : [N, K//16, L]         float8_e4m3fn       (reference layout)
sfa_permuted   : [32,4,ceil(M/128),4,ceil(K//16/4),L]  float8_e4m3fn
sfb1_permuted  : [32,4,ceil(N/128),4,ceil(K//16/4),L]  float8_e4m3fn
sfb2_permuted  : [32,4,ceil(N/128),4,ceil(K//16/4),L]  float8_e4m3fn
c              : [M, N, L]             float16             (output, in-place)
```

**Constraints**: M divisible by `mma_tiler_mn[0]`, N divisible by `mma_tiler_mn[1]`, K divisible by **256**.

**Benchmarks** (speed-of-light targets at 1.5 GHz):
| M   | N    | K    | L | Time (μs) |
|-----|------|------|---|-----------|
| 256 | 4096 | 7168 | 1 | 4.708     |
| 512 | 4096 | 7168 | 1 | 8.714     |
| 256 | 3072 | 4096 | 1 | 2.125     |
| 512 | 3072 | 7168 | 1 | 6.535     |

**Key insight**: A is shared between the two GEMMs — ideal for persistent/fused kernel to avoid re-reading A from DRAM.

---

### Problem 4: `nvfp4_group_gemm` — FP4 Group GEMM

**Operation**: Independent GEMM for each group, variable M/N/K per group, fused dispatch.

**Inputs** (4-tuple of lists):
```
abc_tensors               : list of G tuples (a, b, c) where per group:
    a : [M, K//2, L]        float4_e2m1fn_x2
    b : [N, K//2, L]        float4_e2m1fn_x2
    c : [M, N, L]           float16
sfasfb_tensors            : list of G tuples (sfa, sfb) — reference layout
sfasfb_reordered_tensors  : list of G tuples (sfa_reordered, sfb_reordered) — MMA layout
problem_sizes             : list of G tuples (M, N, K, L)  — L=1 always per group
```

**Constraints**: Per-group M divisible by `mma_tiler_mn[0]`, N by `mma_tiler_mn[1]`, K divisible by **256**. L=1 per group.

**Benchmarks** (speed-of-light targets at 1.5 GHz):
| G | M values                       | N values                       | K values                       | L | Time (μs) |
|---|-------------------------------|-------------------------------|-------------------------------|---|-----------|
| 8 | [80,176,128,72,64,248,96,160]  | [4096]*8                      | [7168]*8                      | 1 | 18.833    |
| 8 | [40,76,168,72,164,148,196,160] | [7168]*8                      | [2048]*8                      | 1 | 10.667    |
| 2 | [192,320]                      | [3072,3072]                   | [4096,4096]                   | 1 | 2.406     |
| 2 | [128,384]                      | [4096,4096]                   | [1536,1536]                   | 1 | 1.525     |

**Relevant sample kernel**: `79d_blackwell_geforce_nvfp4_grouped_gemm` in `sample_kernels/`.

---

## Sample Kernels (Reference Material)

All in `sample_kernels/` — real copies of CUTLASS examples, free to modify:

| File | What it demonstrates |
|------|----------------------|
| `79a_blackwell_geforce_nvfp4_bf16_gemm.cu` | NVFP4→BF16 GEMM, block scaling |
| `79b_blackwell_geforce_nvfp4_nvfp4_gemm.cu` | NVFP4→NVFP4 GEMM |
| `79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu` | Mixed MX-format GEMM |
| `79d_blackwell_geforce_nvfp4_grouped_gemm.cu` | Grouped GEMM (variable M/N/K) |
| `80a_blackwell_geforce_mxfp8_bf16_sparse_gemm.cu` | MXFP8 sparse GEMM |
| `80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm.cu` | NVFP4 sparse GEMM |
| `87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu` | FP8 blockwise scaling GEMM |
| `87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu` | FP8 groupwise scaling GEMM |
| `87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu` | FP8 grouped groupwise GEMM |
| `91_fp4_gemv.cu` | FP4 GEMV (single batch) — structure reference for competition GEMV |

Key CUTLASS headers used across sample kernels:
- `cutlass/gemm/device/gemm_universal_adapter.h` — top-level kernel launcher
- `cutlass/gemm/collective/collective_builder.hpp` — mainloop builder
- `cutlass/epilogue/collective/collective_builder.hpp` — epilogue builder
- `cutlass/detail/sm100_blockscaled_layout.hpp` — block-scaled layout helpers
- `cute/tensor.hpp`, `cute/arch/mma_sm100_desc.hpp` — CuTe primitives

## Kernel Implementation Plan

We write pure C++ / CUTLASS kernels (no Python), one per problem:

| Problem | Target file | Status |
|---------|-------------|--------|
| nvfp4_gemv | `kernels/nvfp4_gemv.cu` | TODO |
| nvfp4_gemm | `kernels/nvfp4_gemm.cu` | TODO |
| nvfp4_dual_gemm | `kernels/nvfp4_dual_gemm.cu` | TODO |
| nvfp4_group_gemm | `kernels/nvfp4_group_gemm.cu` | TODO |

Each kernel will be added as a CMake target following the same pattern as the sample kernels.

## Key Implementation Notes

- Scale factor permuted layout is the MMA-native layout — pass `sfa_permuted`/`sfb_permuted` directly to the CUTLASS collective, not the reference layout
- GEMV N=1 is internally padded to 128 in the reference — the custom kernel does not need this padding
- Dual GEMM: A is read once, reused for both B1 and B2 GEMMs — fuse to avoid extra DRAM traffic
- Group GEMM: L=1 per group always; variable M across groups but fixed N and K in the benchmark configs
- All outputs are FP16; intermediate accumulation in FP32 for dual GEMM (reference uses float32 before casting)
