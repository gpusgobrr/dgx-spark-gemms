# DGX Spark GEMMs

CUTLASS Blackwell GeForce kernel experiments on DGX Spark (SM121a).

## Requirements

- NVIDIA DGX Spark (SM121a, Blackwell GeForce, compute capability 12.1)
- CUDA >= 13.0 (13.1+ required for 80_* sparse and 87_* blockwise/groupwise kernels)
- CMake >= 3.20
- Python 3.12 + uv (for Python environment)

## Setup

```bash
./setup.sh
```

The setup script installs uv, creates a Python 3.12 virtual environment, installs PyTorch with CUDA 13.0 support, and downloads CUTLASS 4.4.1 into `third-party/`.

Custom options:

```bash
./setup.sh -t 4.5.0        # different CUTLASS version
./setup.sh -c 13.1         # different CUDA version
```

## Building

```bash
cmake -B build
cmake --build build

# Or a single target
cmake --build build --target 79b_blackwell_geforce_nvfp4_nvfp4_gemm
```

## Running

```bash
./build/79b_blackwell_geforce_nvfp4_nvfp4_gemm
# Disposition: Passed
# Problem Size: 1024x1024x1024
# Avg runtime: 0.059 ms
# GFLOPS: 36361.5
```

## Kernels

All kernels are in `sample_kernels/` (symlinked from CUTLASS examples) and target SM121a exclusively.

### 79_blackwell_geforce_gemm
| Binary | Description |
|--------|-------------|
| `79a_blackwell_geforce_nvfp4_bf16_gemm` | NVFP4 → BF16 GEMM |
| `79b_blackwell_geforce_nvfp4_nvfp4_gemm` | NVFP4 → NVFP4 GEMM |
| `79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm` | Mixed MXFP8/MXFP6 → BF16 |
| `79d_blackwell_geforce_nvfp4_grouped_gemm` | NVFP4 grouped GEMM |

### 80_blackwell_geforce_sparse_gemm *(requires CUDA 13.1+)*
| Binary | Description |
|--------|-------------|
| `80a_blackwell_geforce_mxfp8_bf16_sparse_gemm` | MXFP8 → BF16 sparse GEMM |
| `80b_blackwell_geforce_nvfp4_nvfp4_sparse_gemm` | NVFP4 sparse GEMM |

### 87_blackwell_geforce_gemm_blockwise *(requires CUDA 13.1+)*
| Binary | Description |
|--------|-------------|
| `87a_blackwell_geforce_fp8_bf16_gemm_blockwise` | FP8 → BF16 blockwise GEMM |
| `87b_blackwell_geforce_fp8_bf16_gemm_groupwise` | FP8 → BF16 groupwise GEMM |
| `87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise` | FP8 → BF16 grouped groupwise GEMM |

### 91_fp4_gemv
| Binary | Description |
|--------|-------------|
| `91_fp4_gemv` | FP4 GEMV (matrix-vector product) |

## Testing

```bash
ctest --test-dir build
ctest --test-dir build -R 79b  # single test
```

## Directory Structure

```
.
├── CMakeLists.txt          # Build configuration (SM121a hardcoded)
├── setup.sh                # Environment setup
├── sample_kernels/         # CUTLASS example kernels (copied from third-party/cutlass/examples/)
└── third-party/
    └── cutlass/            # CUTLASS 4.4.1+
```

## Notes

- Architecture is hardcoded to SM121a — this repo targets DGX Spark only
- `--expt-relaxed-constexpr` is required for the 87_* kernels (CUTLASS blockwise scaling headers)
- NVIDIA driver modules must be loaded before running kernels: `sudo modprobe nvidia`
- Kernel sources in `sample_kernels/` are copied from CUTLASS examples and can be modified freely
