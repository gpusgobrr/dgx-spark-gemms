/*
 * Batched NVFP4 GEMV — Competition kernel for GPU Mode / NVIDIA NVF4 Hackathon
 *
 * Problem: nvfp4_gemv
 *   C[M,1,L] = A[M,K,L] @ B[1,K,L]^T  (block-scaled GEMV, FP16 output)
 *
 * Inputs:
 *   a            : [M, K//2, L]  float4_e2m1fn_x2  (K-major, packed pairs)
 *   b            : [1, K//2, L]  float4_e2m1fn_x2  (K-major)
 *   sfa          : [M, K//16, L] float8_e4m3fn      (block scale factors for A)
 *   sfb          : [1, K//16, L] float8_e4m3fn      (block scale factors for B)
 *   c (output)   : [M, 1, L]    float16
 *
 * Constraints: K divisible by 64, M divisible by 16 (kVectorSize).
 *
 * Benchmark targets (B200 @ 1.5 GHz speed-of-light):
 *   M=7168  K=16384 L=1  →  8.622 μs
 *   M=4096  K=7168  L=8  → 17.275 μs
 *   M=7168  K=2048  L=4  →  4.317 μs
 *
 * Approach: Uses GemvBlockScaled (purpose-built NVFP4 GEMV from 91_fp4_gemv sample)
 *           with a custom FP16 epilogue replacing GemvEpilogueWithScalingFactor.
 *           The custom epilogue bypasses FP4 re-quantization and SFD output,
 *           writing FP16 accumulator results directly.
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "cutlass/util/command_line.h"
// clang-format off
#include "cute/tensor.hpp"
// clang-format on

#include "cute/arch/mma_sm100_desc.hpp"
#include "cute/numeric/numeric_types.hpp"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemv_blockscaled.h"
#include "cutlass/gemm/kernel/gemv_blockscaled.h"
#include "cutlass/epilogue/threadblock/epilogue_with_scaling_factor.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_size.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gemm_complex.h"
#include <cutlass/util/reference/host/gett.hpp>
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/numeric_size.h"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Custom FP16 epilogue for GemvBlockScaled
//
// Replaces GemvEpilogueWithScalingFactor. Satisfies the same interface expected by
// cutlass::gemm::kernel::GemvBlockScaled, but outputs FP16 directly without FP4
// re-quantization or scale factor generation.
//
// Thread layout (RowMajor A path):
//   blockDim = (kThreadsPerRow=8, kThreadCount/kThreadsPerRow=16, 1)
//   threadIdx.x = K-chunk index (0..7)
//   threadIdx.y = row within block (0..15)
//   blockIdx.x  = which block of 16 rows
//   blockIdx.z  = batch index (passed as batch_idx argument)
//
// By the time operator() is called, frag_acc has already been fully reduced across
// K-chunks via warp shuffle in the mainloop. All threadIdx.x values for the same
// threadIdx.y hold the identical final accumulator.
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

template <int kVectorSize_,
          typename ThreadShape_,
          typename ElementCompute_,
          typename ElementAccumulator_,
          typename ElementC_,
          typename ElementD_,
          typename LayoutOutput_>
class GemvEpilogueHalfOutput
{
 public:
  using ThreadShape       = ThreadShape_;
  using ElementCompute    = ElementCompute_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementC          = ElementC_;
  using ElementD          = ElementD_;
  using LayoutOutput      = LayoutOutput_;

  static constexpr int kVectorSize   = kVectorSize_;
  static constexpr int kThreadsPerRow = ThreadShape::kN;   // K-chunks per row = 8
  static constexpr int kThreadsPerCol = ThreadShape::kM;   // rows per block   = 16
  static constexpr int kThreadCount  = kThreadsPerRow * kThreadsPerCol;

  static_assert(cutlass::is_same_v<ElementCompute, float>, "ElementCompute must be float");
  static_assert(cutlass::sizeof_bits<ElementD>::value == 16, "ElementD must be 16-bit (FP16/BF16)");
  static_assert(kThreadsPerCol == 16, "ThreadShape M must be 16");
  static_assert(kThreadsPerRow == 8,  "ThreadShape N must be 8");

  struct Params {
    ElementD*       d_ptr{nullptr};
    ElementC const* c_ptr{nullptr};
    ElementCompute  alpha{1.f};
    ElementCompute  beta{0.f};
    int64_t         batch_stride_d{0};  // elements between batches in D
    int64_t         batch_stride_c{0};  // elements between batches in C
  };

  // No shared memory required: each row is written independently.
  struct SharedStorage {};

 private:
  Params const&  params_;
  SharedStorage& shared_storage_;

 public:
  CUTLASS_HOST_DEVICE
  GemvEpilogueHalfOutput(Params const& params, SharedStorage& shared_storage)
      : params_(params), shared_storage_(shared_storage) {}

  // Called once per thread after K-reduction is complete.
  // frag_acc: fully reduced accumulator for this thread's row (same for all K-chunk threads).
  // frag_c:   C matrix element for this thread's row.
  // batch_idx: current batch (L) index.
  CUTLASS_DEVICE void operator()(ElementAccumulator frag_acc, ElementC frag_c, int batch_idx)
  {
    // Only the first K-chunk thread writes — all K-chunk threads have the same value.
    if (threadIdx.x == 0) {
      float out_val = params_.alpha * float(frag_acc) + params_.beta * float(frag_c);
      ElementD result = ElementD(out_val);

      // Row in the global output: each block covers kVectorSize=16 consecutive rows.
      int row = blockIdx.x * kVectorSize + threadIdx.y;

      // D layout: [M, 1, L] column-major → linear index = batch * M + row
      ElementD* ptr = params_.d_ptr + int64_t(batch_idx) * params_.batch_stride_d + row;
      *ptr = result;
    }
  }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel type definitions
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA   = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_e4m3_t;
using LayoutA    = cutlass::layout::RowMajor;

using ElementB   = cutlass::float_e2m1_t;
using ElementSFB = cutlass::float_e4m3_t;

// The GemvBlockScaled kernel template maps its 4th param to ElementC internally.
// We pass half_t here, so the kernel's ElementC = half_t (ptr_C and ptr_D are half_t*).
using ElementC   = cutlass::half_t;   // kernel's C-bias type (and output type for D pointer)
using ElementD   = cutlass::half_t;   // FP16 output written by our custom epilogue
using LayoutD    = cutlass::layout::ColumnMajor;

using ElementAccumulatorMainloop = cutlass::half_t;
using ElementAccumulator         = float;
using ElementCompute             = float;

static constexpr int kVectorSize        = 16;
static constexpr int kElementsPerAccess = 128 / cutlass::sizeof_bits<ElementA>::value;

using ThreadShape = cutlass::gemm::GemmShape<16, 8>;  // M=16 rows/block, N=8 K-chunks
static_assert(kVectorSize == ThreadShape::kM, "kVectorSize must equal ThreadShape::kM");

// ElementC here is half_t — the kernel casts ptr_C elements to EpilogueOp::ElementC
// before passing frag_c to operator(). Must match the kernel's 4th template param.
using EpilogueOp = cutlass::epilogue::threadblock::GemvEpilogueHalfOutput<
    kVectorSize,
    ThreadShape,
    ElementCompute,
    ElementAccumulator,
    ElementC,   // half_t — matches kernel's internal ElementC
    ElementD,   // half_t — output written by epilogue
    LayoutD>;

using Gemv = cutlass::gemm::device::GemvBlockScaled<
    cutlass::gemm::kernel::GemvBlockScaled<
        ElementA, LayoutA, ElementB, ElementD,
        ElementAccumulatorMainloop, EpilogueOp, kElementsPerAccess>>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

template <typename Element, typename Layout>
bool initialize_tensor(cutlass::TensorView<Element, Layout> view,
                       cutlass::Distribution::Kind dist_kind, uint64_t seed)
{
  if (dist_kind == cutlass::Distribution::Uniform) {
    int bits = cutlass::sizeof_bits<Element>::value;
    double lo = (bits <= 6) ? -2.0 : (bits <= 8 ? -1.0 : -4.0);
    double hi = (bits <= 6) ?  2.0 : (bits <= 8 ?  1.0 :  4.0);
    cutlass::reference::host::TensorFillRandomUniform(view, seed, hi, lo, 0);
  } else if (dist_kind == cutlass::Distribution::AllZeros) {
    cutlass::reference::host::TensorFill(view, Element(0));
  } else {
    CUTLASS_ASSERT(false);
    return false;
  }
  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Testbed
/////////////////////////////////////////////////////////////////////////////////////////////////

using Sm100BlockScaledInputConfig = cutlass::detail::Sm1xxBlockScaledConfig<kVectorSize>;
using Blk_MN_Input = typename Sm100BlockScaledInputConfig::Blk_MN;
using Blk_SF_Input = typename Sm100BlockScaledInputConfig::Blk_SF;
using SfAtom_Input = typename Sm100BlockScaledInputConfig::SfAtom;

struct Testbed {
  // Tensors
  cutlass::HostTensor<ElementA,   LayoutA>                             tensor_A;
  cutlass::HostTensor<ElementSFA, LayoutA>                             tensor_SFA;
  cutlass::HostTensor<ElementB,   cutlass::layout::ColumnMajor>        tensor_B;
  cutlass::HostTensor<ElementSFB, cutlass::layout::ColumnMajor>        tensor_SFB;
  cutlass::HostTensor<ElementC,   LayoutD>                             tensor_C;  // C bias (zeroed)
  cutlass::HostTensor<ElementD,   LayoutD>                             tensor_D;
  cutlass::HostTensor<ElementD,   LayoutD>                             reference_D;

  bool initialize(int m, int k, int batch, uint64_t seed = 42)
  {
    const int n = 1;
    auto k_blks = cutlass::ceil_div(k, cute::size<1>(shape(SfAtom_Input{})));
    auto m_blks = cutlass::ceil_div(m, Blk_MN_Input{});
    auto n_blks = cutlass::ceil_div(n, Blk_MN_Input{});

    // A: [batch*M, K] row major
    tensor_A.resize({batch * m, k});
    // SFA: blocked layout matching CUTLASS GEMV expectations
    auto sfa_coord = cutlass::make_Coord(m_blks * Blk_MN_Input{} * batch,
                                         k_blks * Blk_SF_Input{});
    auto sfa_layout = cutlass::layout::Affine2Layout_Factory<LayoutA>::layout_factory(
        sfa_coord, typename LayoutA::Stride{});
    tensor_SFA.resize(sfa_coord, sfa_layout);

    // B: [batch*K, 1] column major
    tensor_B.resize({batch * k, n});
    auto sfb_coord = cutlass::make_Coord(n_blks * Blk_MN_Input{} * batch,
                                         k_blks * Blk_SF_Input{});
    auto sfb_layout = cutlass::layout::Affine2Layout_Factory<cutlass::layout::ColumnMajor>::layout_factory(
        sfb_coord, cutlass::layout::ColumnMajor::Stride{});
    tensor_SFB.resize(sfb_coord, sfb_layout);

    // C/D/reference: [batch*M, 1] column major
    tensor_C.resize({batch * m, n});
    tensor_D.resize({batch * m, n});
    reference_D.resize({batch * m, n});

    initialize_tensor(tensor_A.host_view(),   cutlass::Distribution::Uniform,  seed + 1);
    initialize_tensor(tensor_SFA.host_view(), cutlass::Distribution::Uniform,  seed + 2);
    initialize_tensor(tensor_B.host_view(),   cutlass::Distribution::Uniform,  seed + 3);
    initialize_tensor(tensor_SFB.host_view(), cutlass::Distribution::Uniform,  seed + 4);
    initialize_tensor(tensor_C.host_view(),   cutlass::Distribution::AllZeros, seed + 5);
    initialize_tensor(tensor_D.host_view(),   cutlass::Distribution::AllZeros, seed + 6);

    tensor_A.sync_device();
    tensor_SFA.sync_device();
    tensor_B.sync_device();
    tensor_SFB.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();

    return true;
  }

  typename Gemv::Arguments make_arguments(int m, int k, int batch,
                                           ElementCompute alpha, ElementCompute beta)
  {
    const int n = 1;
    auto k_blks = cutlass::ceil_div(k, cute::size<1>(shape(SfAtom_Input{})));
    auto m_blks = cutlass::ceil_div(m, Blk_MN_Input{});
    auto n_blks = cutlass::ceil_div(n, Blk_MN_Input{});

    int batch_stride_SFA = m_blks * Blk_MN_Input{} * k_blks * Blk_SF_Input{};
    int batch_stride_SFB = n_blks * Blk_MN_Input{} * k_blks * Blk_SF_Input{};

    typename EpilogueOp::Params epi_params;
    epi_params.d_ptr          = tensor_D.device_data();
    epi_params.c_ptr          = tensor_C.device_data();  // kernel always reads ptr_C
    epi_params.alpha          = alpha;
    epi_params.beta           = beta;
    epi_params.batch_stride_d = m;  // D is [M,1,L] → M elements per batch
    epi_params.batch_stride_c = m;

    typename Gemv::Arguments arguments{
        cutlass::MatrixCoord{m, k},
        batch,
        epi_params,
        tensor_A.device_ref(),
        tensor_B.device_data(),
        tensor_C.device_data(),        // ptr_C (kernel always reads this regardless of beta)
        tensor_D.device_data(),
        tensor_SFA.device_data(),
        tensor_SFB.device_data(),
        k,                             // stride_A
        m * k,                         // batch_stride_A
        k,                             // batch_stride_B
        m,                             // batch_stride_C
        m,                             // batch_stride_D
        batch_stride_SFA,
        batch_stride_SFB,
        0                              // batch_stride_SFD (unused)
    };

    return arguments;
  }

  bool run(int m, int k, int batch, ElementCompute alpha, ElementCompute beta, int iterations,
           bool do_verify)
  {
    initialize(m, k, batch);

    Gemv gemv;
    auto arguments = make_arguments(m, k, batch, alpha, beta);

    cutlass::Status status = gemv.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "can_implement() failed: " << cutlass::cutlassGetStatusString(status) << "\n";
      return false;
    }

    size_t ws = Gemv::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);

    status = gemv.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "initialize() failed: " << cutlass::cutlassGetStatusString(status) << "\n";
      return false;
    }

    // Correctness check
    if (do_verify) {
      status = gemv();
      if (status != cutlass::Status::kSuccess) {
        std::cerr << "Kernel failed: " << cutlass::cutlassGetStatusString(status) << "\n";
        return false;
      }

      // Host reference (GETT)
      using ProblemShapeType = cute::Shape<int,int,int,int>;
      auto problem_shape = ProblemShapeType{m, 1, k, batch};

      using StrideARef = cutlass::gemm::TagToStrideA_t<LayoutA>;
      using StrideBRef = cutlass::gemm::TagToStrideB_t<cutlass::layout::ColumnMajor>;
      using StrideDRef = cutlass::gemm::TagToStrideC_t<LayoutD>;

      StrideARef stride_a = cutlass::make_cute_packed_stride(StrideARef{}, cute::make_shape(m, k, batch));
      StrideBRef stride_b = cutlass::make_cute_packed_stride(StrideBRef{}, cute::make_shape(1, k, batch));
      StrideDRef stride_d = cutlass::make_cute_packed_stride(StrideDRef{}, cute::make_shape(m, 1, batch));

      auto layout_sfa = Sm100BlockScaledInputConfig::tile_atom_to_shape_SFA(problem_shape);
      auto layout_sfb = Sm100BlockScaledInputConfig::tile_atom_to_shape_SFB(problem_shape);

      auto tensor_A_cute   = make_tensor(make_iterator(tensor_A.host_data()),
                                         make_layout(make_shape(m, k, batch), stride_a));
      auto tensor_SFA_cute = make_tensor(tensor_SFA.host_data(), layout_sfa);
      auto tensor_B_cute   = make_tensor(make_iterator(tensor_B.host_data()),
                                         make_layout(make_shape(1, k, batch), stride_b));
      auto tensor_SFB_cute = make_tensor(tensor_SFB.host_data(), layout_sfb);
      auto ref_D_cute      = make_tensor(make_iterator(reference_D.host_data()),
                                         make_layout(make_shape(m, 1, batch), stride_d));

      // Dummy C (beta=0)
      cutlass::HostTensor<ElementD, LayoutD> dummy_C;
      dummy_C.resize({m, 1});
      initialize_tensor(dummy_C.host_view(), cutlass::Distribution::AllZeros, 0);
      auto tensor_C_cute = make_tensor(make_iterator(dummy_C.host_data()),
                                        make_layout(make_shape(m, 1, 1), stride_d));

      cutlass::reference::host::GettBlockScalingMainloopParams<
          ElementAccumulator,
          decltype(tensor_A_cute), decltype(tensor_SFA_cute),
          decltype(tensor_B_cute), decltype(tensor_SFB_cute)
        > mainloop{tensor_A_cute, tensor_SFA_cute, tensor_B_cute, tensor_SFB_cute};

      cutlass::reference::host::GettBlockScalingEpilogueParams<
          ElementAccumulator, ElementAccumulator, ElementAccumulator,
          decltype(tensor_C_cute), decltype(ref_D_cute)
        > epilogue{alpha, beta, tensor_C_cute, ref_D_cute};

      cutlass::reference::host::Gemm3x(mainloop, epilogue);

      tensor_D.sync_host();
      bool passed = cutlass::reference::host::TensorEquals(
          reference_D.host_view(), tensor_D.host_view());
      std::cout << "Verification: " << (passed ? "PASSED" : "FAILED") << "\n";
      if (!passed) return false;
    }

    // Profiling
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    for (int i = 0; i < 5; ++i) gemv();
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    for (int i = 0; i < iterations; ++i) gemv();
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, t0, t1);
    double avg_us = double(elapsed_ms) / iterations * 1000.0;

    uint64_t flops = uint64_t(2) * m * k * batch;
    double tflops  = double(flops) / 1e12 / (avg_us * 1e-6);

    std::cout << "Problem: M=" << m << " K=" << k << " L=" << batch << " N=1\n";
    std::cout << "Avg runtime: " << avg_us << " us\n";
    std::cout << "TFLOPS: " << tflops << "\n";

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Options
/////////////////////////////////////////////////////////////////////////////////////////////////

struct Options {
  bool help       = false;
  int  m          = 7168;
  int  k          = 2048;
  int  l          = 1;
  float alpha     = 1.f;
  float beta      = 0.f;
  int  iterations = 100;
  bool verify     = true;

  void parse(int argc, char const** args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta",  beta,  0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verify, true);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "nvfp4_gemv — Batched NVFP4 block-scaled GEMV with FP16 output\n\n"
        << "  C[M,1,L] = A[M,K,L] @ B[1,K,L]^T\n\n"
        << "Options:\n"
        << "  --m=<int>          M dimension (default: 7168)\n"
        << "  --k=<int>          K dimension (default: 2048)\n"
        << "  --l=<int>          Batch size L (default: 1)\n"
        << "  --alpha=<f32>      Scale factor (default: 1.0)\n"
        << "  --beta=<f32>       Bias scale   (default: 0.0)\n"
        << "  --iterations=<int> Profiling iterations (default: 100)\n"
        << "  --verify=<bool>    Run host verification (default: true)\n\n"
        << "Competition benchmarks:\n"
        << "  --m=7168 --k=16384 --l=1   (target: 8.622 us)\n"
        << "  --m=4096 --k=7168  --l=8   (target: 17.275 us)\n"
        << "  --m=7168 --k=2048  --l=4   (target: 4.317 us)\n\n";
    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Main
/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv)
{
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  Options opt;
  opt.parse(argc, argv);

  if (opt.help) {
    opt.print_usage(std::cout);
    return 0;
  }

  Testbed testbed;
  bool ok = testbed.run(opt.m, opt.k, opt.l, opt.alpha, opt.beta, opt.iterations, opt.verify);
  return ok ? 0 : 1;

#else
  std::cerr << "Unsupported: CUTLASS_ARCH_MMA_SM100_SUPPORTED not defined\n";
  return 1;
#endif
}
