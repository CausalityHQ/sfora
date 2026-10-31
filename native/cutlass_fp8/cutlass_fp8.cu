// CUTLASS sm_120/sm_121 FP8 GEMMs with fused epilogues for the So400m fused FP8 encoder (inference only).
// Plain C ABI (ctypes); all pointers are device pointers, all matrices row-major, weights (N, K) row-major.
//   linear      : D(fp16) = fp16(acc * sa[m] * sw[n] + bias[n])                      (dense e4m3, per-row/col scales)
//   linear_res  : D(fp16) = C + fp16(acc * sa[m] * sw[n] + bias[n])                  (o-proj + residual add)
//   fc1_gelu_mx : D(e4m3) + SF(ue8m0, 1x32, 128x4-swizzled) = MXFP8(fp16 GELU_tanh(fp16(acc*sa*sw + bias)))
//   mx_res      : D(fp16) = C + fp16(acc_mx * r[n] + bias[n])                        (MXFP8 fc2 + residual add)
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_visitor_store_tma_warpspecialized.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;
namespace fu = cutlass::epilogue::fusion;
using half_t = cutlass::half_t;
using e4m3 = cutlass::float_e4m3_t;

using Cluster = Shape<_1, _1, _1>;
using EpiTile = Shape<_64, _32>;  // = the sm120 auto choice; fixed so the SF-store node can name it
constexpr int kFrag = 4;          // sm120 epilogue fragment size
constexpr auto kRN = cutlass::FloatRoundStyle::round_to_nearest;

// tanh-GELU with the Triton formula (exact exp, not tanh.approx), result rounded to fp16 like the served module.
template <class T> struct GeluTanhF16;
template <int N> struct GeluTanhF16<cutlass::Array<float, N>> {
  CUTLASS_DEVICE cutlass::Array<float, N> operator()(cutlass::Array<float, N> const& x) const {
    cutlass::Array<float, N> y;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      float v = x[i], u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
      y[i] = __half2float(__float2half_rn(0.5f * v * (2.f - 2.f / (1.f + __expf(2.f * u)))));
    }
    return y;
  }
};

template <class T> using RowScale = fu::Sm90ColBroadcast<0, T, float>;       // per-M (activation row scale)
template <class T> using ColScale = fu::Sm90RowBroadcast<0, T, float>;       // per-N (weight scale / MX residual r)
template <class T> using BiasN = fu::Sm90RowBroadcast<0, T, half_t, float>;  // per-N bias
using Acc = fu::Sm90AccFetch;
template <class T> using Scaled = fu::Sm90EVT<fu::Sm90Compute<cutlass::multiplies, float, float, kRN>, RowScale<T>, Acc>;
template <class T> using Lin =
    fu::Sm90EVT<fu::Sm90Compute<cutlass::homogeneous_multiply_add, half_t, float, kRN>, ColScale<T>, Scaled<T>, BiasN<T>>;
template <class T> using LinRes = fu::Sm90EVT<fu::Sm90Compute<cutlass::plus, half_t, half_t, kRN>, fu::Sm90SrcFetch<half_t>, Lin<T>>;
template <class T> using GeluMx = fu::Sm90EVT<
    fu::Sm120BlockScaleFactorRowStore<32, EpiTile, T, kFrag, e4m3, float, cutlass::float_ue8m0_t>,
    fu::Sm90EVT<fu::Sm90Compute<GeluTanhF16, float, float, kRN>, Lin<T>>>;
template <class T> using LinMx =
    fu::Sm90EVT<fu::Sm90Compute<cutlass::homogeneous_multiply_add, half_t, float, kRN>, ColScale<T>, Acc, BiasN<T>>;
template <class T> using LinMxRes = fu::Sm90EVT<fu::Sm90Compute<cutlass::plus, half_t, half_t, kRN>, fu::Sm90SrcFetch<half_t>, LinMx<T>>;

// Pingpong: two consumer warpgroups on alternating tiles, so one's epilogue overlaps the other's mainloop.
template <bool PP> using KSched = std::conditional_t<PP, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                                     cutlass::gemm::collective::KernelScheduleAuto>;
template <bool PP> using ESched = std::conditional_t<PP, cutlass::epilogue::TmaWarpSpecialized,
                                                     cutlass::epilogue::collective::EpilogueScheduleAuto>;

template <class OpClass, class ElemAB, class ElemC, class ElemD, template <class> class Evt, class CtaTile, bool PP>
struct G {
  static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElemD>::value;
  static constexpr int AlignC = cute::is_void_v<ElemC> ? AlignD : 128 / cutlass::sizeof_bits<ElemC>::value;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, CtaTile, Cluster, EpiTile, float, float,
      ElemC, cutlass::layout::RowMajor, AlignC, ElemD, cutlass::layout::RowMajor, AlignD, ESched<PP>, Evt<CtaTile>>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, OpClass, ElemAB, cutlass::layout::RowMajor, 16, ElemAB, cutlass::layout::ColumnMajor, 16,
      float, CtaTile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      KSched<PP>>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using SA = typename Kernel::StrideA;
  using SB = typename Kernel::StrideB;
  using SC = typename Kernel::StrideC;
  using SD = typename Kernel::StrideD;
};

// Per-GEMM schedule / tile (override all four with -D for tuning builds): PP = pingpong, TM/TN/TK = CTA tile.
// Defaults = fastest of tune_gemms.py on GB10 (pingpong beats cooperative by 0.8 ms on fc1: GELU+MX epilogue overlap;
// 128x256 tiles spill, 128x64 / 64x256 slower, K=64 fails to build for the MX GEMM).
#ifndef LIN_PP
#define LIN_PP 1
#define LIN_TM 128
#define LIN_TN 128
#define LIN_TK 64
#endif
#ifndef O_PP
#define O_PP 1
#define O_TM 128
#define O_TN 128
#define O_TK 128
#endif
#ifndef FC1_PP
#define FC1_PP 1
#define FC1_TM 128
#define FC1_TN 128
#define FC1_TK 128
#endif
#ifndef FC2_PP
#define FC2_PP 1
#define FC2_TM 128
#define FC2_TN 128
#define FC2_TK 128
#endif
#define TILE(P) Shape<Int<P##_TM>, Int<P##_TN>, Int<P##_TK>>

using Dense = cutlass::arch::OpClassTensorOp;
using Blk = cutlass::arch::OpClassBlockScaledTensorOp;
using GLin = G<Dense, e4m3, void, half_t, Lin, TILE(LIN), LIN_PP>;
using GLinRes = G<Dense, e4m3, half_t, half_t, LinRes, TILE(O), O_PP>;
using GFc1 = G<Dense, e4m3, void, e4m3, GeluMx, TILE(FC1), FC1_PP>;
using GMxRes = G<Blk, cutlass::mx_float8_t<e4m3>, half_t, half_t, LinMxRes, TILE(FC2), FC2_PP>;

static void* workspace(size_t n) {
  static void* p = nullptr;
  static size_t cap = 0;
  if (n > cap) {
    if (p) {
      cudaDeviceSynchronize();  // only on growth (first calls); never frees under an in-flight kernel
      cudaFree(p);
    }
    cudaMalloc(&p, n);
    cap = n;
  }
  return p;
}

static cutlass::KernelHardwareInfo hw() {
  static cutlass::KernelHardwareInfo h = [] {
    cutlass::KernelHardwareInfo x;
    cudaGetDevice(&x.device_id);
    x.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(x.device_id);
    return x;
  }();
  return h;
}

template <class Gemm>
static int run(typename Gemm::Arguments const& args, cudaStream_t s) {
  Gemm g;
  if (g.can_implement(args) != cutlass::Status::kSuccess) return 1;
  if (g.initialize(args, workspace(Gemm::get_workspace_size(args)), s) != cutlass::Status::kSuccess) return 2;
  if (g.run(s) != cutlass::Status::kSuccess) return 3;
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

template <class T>
static typename Lin<T>::Arguments lin_args(const float* sa, const float* sw, const void* bias) {
  return {{sw}, {{sa}, {}, {}}, {static_cast<const half_t*>(bias)}, {}};
}

template <class GG, class FusionArgs>
static typename GG::Gemm::Arguments dense_args(const void* A, const void* W, FusionArgs f, const void* C, void* D,
                                               int M, int N, int K) {
  auto sa = cutlass::make_cute_packed_stride(typename GG::SA{}, {M, K, 1});
  auto sb = cutlass::make_cute_packed_stride(typename GG::SB{}, {N, K, 1});
  auto sc = cutlass::make_cute_packed_stride(typename GG::SC{}, {M, N, 1});
  auto sd = cutlass::make_cute_packed_stride(typename GG::SD{}, {M, N, 1});
  using EC = typename GG::Gemm::GemmKernel::CollectiveEpilogue;
  return {cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
          {static_cast<const e4m3*>(A), sa, static_cast<const e4m3*>(W), sb},
          {f, static_cast<typename EC::ElementC const*>(C), sc, static_cast<typename EC::ElementD*>(D), sd}, hw()};
}

extern "C" int cf8_linear(const void* A, const void* W, const float* sa, const float* sw, const void* bias, void* D,
                          int M, int N, int K, cudaStream_t s) {
  return run<GLin::Gemm>(dense_args<GLin>(A, W, lin_args<TILE(LIN)>(sa, sw, bias), nullptr, D, M, N, K), s);
}

extern "C" int cf8_linear_res(const void* A, const void* W, const float* sa, const float* sw, const void* bias,
                              const void* C, void* D, int M, int N, int K, cudaStream_t s) {
  typename LinRes<TILE(O)>::Arguments f{{}, lin_args<TILE(O)>(sa, sw, bias), {}};
  return run<GLinRes::Gemm>(dense_args<GLinRes>(A, W, f, C, D, M, N, K), s);
}

extern "C" int cf8_fc1_gelu_mx(const void* A, const void* W, const float* sa, const float* sw, const void* bias,
                               void* D, void* SF, const float* norm_one, int M, int N, int K, cudaStream_t s) {
  typename GeluMx<TILE(FC1)>::Arguments f{{lin_args<TILE(FC1)>(sa, sw, bias), {}},
                               {static_cast<cutlass::float_ue8m0_t*>(SF), norm_one, {}}};
  return run<GFc1::Gemm>(dense_args<GFc1>(A, W, f, nullptr, D, M, N, K), s);
}

extern "C" int cf8_mx_res(const void* A, const void* SFA, const void* W, const void* SFW, const float* r,
                          const void* bias, const void* C, void* D, int M, int N, int K, cudaStream_t s) {
  using Cfg = typename GMxRes::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto sa = cutlass::make_cute_packed_stride(typename GMxRes::SA{}, {M, K, 1});
  auto sb = cutlass::make_cute_packed_stride(typename GMxRes::SB{}, {N, K, 1});
  auto sc = cutlass::make_cute_packed_stride(typename GMxRes::SC{}, {M, N, 1});
  auto lsfa = Cfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto lsfb = Cfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  using SFT = cutlass::float_ue8m0_t;
  typename LinMxRes<TILE(FC2)>::Arguments f{{}, {{r}, {}, {static_cast<const half_t*>(bias)}, {}}, {}};
  typename GMxRes::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K, 1},
      {static_cast<const e4m3*>(A), sa, static_cast<const e4m3*>(W), sb, static_cast<const SFT*>(SFA), lsfa,
       static_cast<const SFT*>(SFW), lsfb},
      {f, static_cast<const half_t*>(C), sc, static_cast<half_t*>(D), sc}, hw()};
  return run<GMxRes::Gemm>(args, s);
}
