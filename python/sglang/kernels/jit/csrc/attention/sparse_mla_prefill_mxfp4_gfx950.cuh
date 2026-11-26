// gfx950 DSA sparse MLA prefill with 400-byte MXFP4 KV rows:
// [256 packed E2M1 NoPE bytes | 16 E8M0 scales | 64 BF16 RoPE values].
// The CTA, softmax, DMA schedule and split output layout match the FP8 prefill kernel.
#pragma once

#include "sparse_mla_prefill_gfx950.cuh"

namespace sglang::sparse_mla_prefill_mxfp4_gfx950 {

namespace fp8 = sparse_mla_prefill_gfx950;
using fp8::bf16x2;
using fp8::bf16x8;
using fp8::floatx4;
using fp8::intx2;
using fp8::kBufferConfig;
using fp8::kDv;
using fp8::kKeys;
using fp8::kRowsPerWave;
using fp8::kStages;
using fp8::kThreads;
using fp8::kWaves;
using fp8::kXcds;
using fp8::lds_barrier;
using fp8::lds_intx2;
using fp8::mfma;
using fp8::Params;
using fp8::xmax16;
using fp8::xmax32;
using fp8::xsum16;
using fp8::xsum32;

constexpr int kRow = 400;
constexpr int kScaleOff = 256;
constexpr int kRopeOff = 272;
constexpr int kRowChunks = 25;
constexpr int kTileBytes = kKeys * kRow;
static_assert(kRowChunks * 16 == kRow && kKeys == 32 && kWaves == 4 && kStages == 2);

SGL_DEVICE float e8m0(uint32_t scales, int blk) {
  return __uint_as_float(((scales >> (8 * blk)) & 0xffu) << 23);
}

SGL_DEVICE bf16x8 cvt_fp4(uint32_t v, float scale) {
  const bf16x2 a = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(v, scale, 0);
  const bf16x2 b = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(v, scale, 1);
  const bf16x2 c = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(v, scale, 2);
  const bf16x2 d = __builtin_amdgcn_cvt_scalef32_pk_bf16_fp4(v, scale, 3);
  return bf16x8{a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1]};
}

// kStages tiles of KV rows live in LDS; wave w fetches rows [8w, 8w + 8) of each tile with
// LDS DMA (lanes 0..24 move one 16-B chunk each), addressed from wave-uniform indices.
template <bool kSplit>
__global__ __launch_bounds__(kThreads) void sparse_mla_prefill_mxfp4_kernel(const Params p, uint32_t kv_stride_b) {
  __shared__ __attribute__((aligned(16))) uint8_t s_kv[kStages][kTileBytes];
  __shared__ __attribute__((aligned(16))) float s_mask[kStages][kKeys];  // 0 or -inf per key
  __shared__ floatx4 s_red[kWaves][2][64];

  // Blocks go round-robin over the 8 XCDs; give each XCD a contiguous token range so that
  // neighbouring tokens, whose top-k sets overlap, share that XCD's L2.
  int t;
  {
    const int T = static_cast<int>(gridDim.x);
    const int q = T / kXcds;
    const int b = static_cast<int>(blockIdx.x);
    t = b < q * kXcds ? (b % kXcds) * q + b / kXcds : b;
  }
  const int tid = threadIdx.x;
  // Each wave owns 128 NoPE dims (four MXFP4 scale blocks). Keep w wave-uniform in an
  // SGPR: MFMA ignores EXEC, so a divergent branch could run both sides' MFMAs.
  const int w = __builtin_amdgcn_readfirstlane(tid >> 6);
  const int l = tid & 63;
  const int n = l & 15;  // head
  const int g = l >> 4;
  // this CTA's key tiles: [j0, j0 + ntiles)
  int j0 = 0;
  int ntiles = (p.topk + kKeys - 1) / kKeys;
  if constexpr (kSplit) {
    j0 = static_cast<int>(blockIdx.y) * p.tiles_per_split;
    ntiles = max(0, min(ntiles - j0, p.tiles_per_split));
  }
  const int32_t* idx_row = p.idx + static_cast<int64_t>(t) * p.topk;

  // Q^T as the QK B operand: col = head n, k = dims (g * 8 + j) of each 32-dim step.
  const bf16x8 zero8 = {};
  const bool head_ok = n < p.H;
  bf16x8 qn[4];
  bf16x8 qr = zero8;
  const int rks = w & 1;   // this wave's RoPE k-step
  const int rnb = w >> 1;  // ... and key half
  {
    const __bf16* qp = p.q_nope + t * p.sqn_t + n * p.sqn_h + w * 128 + g * 8;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks)
      qn[ks] = head_ok ? *reinterpret_cast<const bf16x8*>(qp + ks * 32) : zero8;
    if (head_ok) qr = *reinterpret_cast<const bf16x8*>(p.q_rope + t * p.sqr_t + n * p.sqr_h + rks * 32 + g * 8);
  }

  // Lanes l hold the index of row 8w + (l & 7) of tile j.
  // Branch-free (clamped address, then masked) so no load result merges at a join, which
  // would make the compiler wait for every outstanding load.
  const uint32_t kv_rows = p.kv_bytes / kv_stride_b;
  auto load_idx = [&](int j) -> int {
    const int k = (j0 + j) * kKeys + kRowsPerWave * w + (l & 7);
    const int v = idx_row[min(k, p.topk - 1)];
    return k < p.topk ? v : -1;
  };
  // Invalid rows read through an empty descriptor: out-of-range buffer loads return zeros.
  const __amdgpu_buffer_rsrc_t rsrc_kv =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t*>(p.kv), 0, static_cast<int>(p.kv_bytes), kBufferConfig);
  const __amdgpu_buffer_rsrc_t rsrc_none =
      __builtin_amdgcn_make_buffer_rsrc(const_cast<uint8_t*>(p.kv), 0, 0, kBufferConfig);
  auto issue_kv = [&](int stage, int vidx) {
    // Check bounds at use, not on the prefetched VMEM result.
    if (l < kRowsPerWave)
      s_mask[stage][kRowsPerWave * w + l] = static_cast<uint32_t>(vidx) < kv_rows ? 0.0f : -INFINITY;
    if (l < kRowChunks) {
#pragma unroll
      for (int r = 0; r < kRowsPerWave; ++r) {
        const uint32_t id = __builtin_amdgcn_readlane(vidx, r);
        auto* dst = (__attribute__((address_space(3))) void*)&s_kv[stage][(kRowsPerWave * w + r) * kRow];
        if (id < kv_rows)
          __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc_kv, dst, 16, l * 16, id * kv_stride_b, 0, 0);
        else
          __builtin_amdgcn_raw_ptr_buffer_load_lds(rsrc_none, dst, 16, l * 16, 0, 0, 0);
      }
    }
  };

  // VMEM order per iteration i: 8 DMAs of tile i+S-1, then the index load of tile i+2S-1.
  // vq[r] = indices of tile j+S-1+r at the top of iteration j, so the indices a DMA needs
  // were loaded before the DMAs of the tile we wait for.
  int vq[kStages];
#pragma unroll
  for (int j = 0; j < kStages - 1; ++j)
    if (j < ntiles) issue_kv(j, load_idx(j));
#pragma unroll
  for (int r = 0; r < kStages; ++r)
    vq[r] = load_idx(kStages - 1 + r);

  // Online softmax state for head n (identical in all 4 waves), log2 domain.
  float m_i = -INFINITY;
  float l_i = 0.0f;  // this lane's partial row sum; summed over the 4 lane groups at the end
  floatx4 acc[8];
#pragma unroll
  for (int nb = 0; nb < 8; ++nb)
    acc[nb] = floatx4{0, 0, 0, 0};

  // PV A through tr4: each 16-lane group transposes a 16x16 nibble matrix.
  // Input lane s supplies key K_g[s % 8], where K_g = {4g..4g+3, 16+4g..16+4g+3}
  // is the QK C layout. Lanes 0..7 read the first 16 dims and lanes 8..15 read the next
  // 16 dims, so lane n receives two dwords: K_g at dim n and at dim n+16, respectively.
  const int v_j = n & 7;
  const int v_key = v_j < 4 ? g * 4 + v_j : 16 + g * 4 + (v_j - 4);
  const int v_off = v_key * kRow + w * 64 + (n >> 3) * 8;

  for (int j = 0; j < ntiles; ++j) {
    const int stage = j % kStages;
    // Tile j's DMAs must have landed; only one index load was issued after them.
    asm volatile("s_waitcnt vmcnt(1)" ::: "memory");
    lds_barrier();  // tile j visible to all waves; everyone is done with tile j-1
    if (j + kStages - 1 < ntiles) issue_kv((j + kStages - 1) % kStages, vq[0]);
#pragma unroll
    for (int r = 0; r + 1 < kStages; ++r)
      vq[r] = vq[r + 1];
    vq[kStages - 1] = load_idx(j + 2 * kStages - 1);

    const uint8_t* kvs = s_kv[stage];

    // ---- QK partial over this wave's 128 NoPE dims: S^T[key, head] ----
    floatx4 c0 = {0, 0, 0, 0}, c1 = {0, 0, 0, 0};
    {
      const uint8_t* r0 = kvs + n * kRow;
      const uint8_t* r1 = r0 + 16 * kRow;
      const uint32_t sc0 = *reinterpret_cast<const uint32_t*>(r0 + kScaleOff + 4 * w);
      const uint32_t sc1 = *reinterpret_cast<const uint32_t*>(r1 + kScaleOff + 4 * w);
#pragma unroll
      for (int blk = 0; blk < 4; ++blk) {
        const int off = w * 64 + blk * 16 + g * 4;
        c0 = mfma(cvt_fp4(*reinterpret_cast<const uint32_t*>(r0 + off), e8m0(sc0, blk)), qn[blk], c0);
        c1 = mfma(cvt_fp4(*reinterpret_cast<const uint32_t*>(r1 + off), e8m0(sc1, blk)), qn[blk], c1);
      }
    }
    {
      const bf16x8 kr =
          *reinterpret_cast<const bf16x8*>(kvs + (rnb * 16 + n) * kRow + kRopeOff + (rks * 32 + g * 8) * 2);
      if (rnb == 0)
        c0 = mfma(kr, qr, c0);
      else
        c1 = mfma(kr, qr, c1);
    }
    s_red[w][0][l] = c0;
    s_red[w][1][l] = c1;
    lds_barrier();
    floatx4 s0 = s_red[0][0][l], s1 = s_red[0][1][l];
#pragma unroll
    for (int ww = 1; ww < kWaves; ++ww) {
      s0 += s_red[ww][0][l];
      s1 += s_red[ww][1][l];
    }

    // ---- online softmax over the tile's 32 keys for head n ----
    const floatx4 mk0 = *reinterpret_cast<const floatx4*>(&s_mask[stage][g * 4]);
    const floatx4 mk1 = *reinterpret_cast<const floatx4*>(&s_mask[stage][16 + g * 4]);
    float sv[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      sv[i] = fmaf(s0[i], p.qk_scale, mk0[i]);
      sv[4 + i] = fmaf(s1[i], p.qk_scale, mk1[i]);
    }
    float mloc =
        fmaxf(fmaxf(fmaxf(sv[0], sv[1]), fmaxf(sv[2], sv[3])), fmaxf(fmaxf(sv[4], sv[5]), fmaxf(sv[6], sv[7])));
    mloc = xmax16(xmax32(mloc));
    // Lazy rescale: keep the running max unless some head's max grew by more than 2^8, so
    // p <= 256 and the accumulator is rescaled only on the rare tiles that move it.
    if (__builtin_amdgcn_ballot_w64(mloc > m_i + 8.0f) != 0) {
      const float m_new = fmaxf(m_i, mloc);
      const float alpha = __builtin_amdgcn_exp2f(m_i - (m_new == -INFINITY ? 0.0f : m_new));
      l_i *= alpha;
#pragma unroll
      for (int nb = 0; nb < 8; ++nb)
        acc[nb] *= alpha;
      m_i = m_new;
    }
    const float m_use = m_i == -INFINITY ? 0.0f : m_i;
    float psum = 0.0f;
    float prob[8];
    uint32_t scale[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      prob[i] = __builtin_amdgcn_exp2f(sv[i] - m_use);
      prob[4 + i] = __builtin_amdgcn_exp2f(sv[4 + i] - m_use);
      psum += prob[i] + prob[4 + i];
      scale[i] = *reinterpret_cast<const uint32_t*>(kvs + (g * 4 + i) * kRow + kScaleOff + 4 * w);
      scale[4 + i] = *reinterpret_cast<const uint32_t*>(kvs + (16 + g * 4 + i) * kRow + kScaleOff + 4 * w);
    }
    l_i += psum;

    // ---- PV over this wave's 128 output dims: O^T[dim, head] ----
    // Each 32-dim block has its own per-key E8M0 scale. Fold it into P, leaving V at unit scale.
#pragma unroll
    for (int blk = 0; blk < 4; ++blk) {
      bf16x8 pb;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        pb[i] = static_cast<__bf16>(prob[i] * e8m0(scale[i], blk));
      const intx2 v = __builtin_amdgcn_ds_read_tr4_b64_v2i32((lds_intx2*)(kvs + v_off + blk * 16));
      acc[2 * blk] = mfma(cvt_fp4(v[0], 1.0f), pb, acc[2 * blk]);
      acc[2 * blk + 1] = mfma(cvt_fp4(v[1], 1.0f), pb, acc[2 * blk + 1]);
    }
  }

  const float l_tot = xsum16(xsum32(l_i));
  if (!head_ok) return;
  if constexpr (kSplit) {
    const int64_t ps = (static_cast<int64_t>(t) * gridDim.y + blockIdx.y) * p.H + n;
    float* ap = p.part_acc + ps * kDv + w * 128 + g * 4;
#pragma unroll
    for (int nb = 0; nb < 8; ++nb)
      *reinterpret_cast<floatx4*>(ap + nb * 16) = acc[nb];
    if (w == 0 && g == 0) {
      p.part_m[ps] = m_i;
      p.part_l[ps] = l_tot;
    }
    return;
  }
  const float inv = l_tot > 0.0f ? 1.0f / l_tot : 0.0f;
  __bf16* op = p.out + (static_cast<int64_t>(t) * p.H + n) * kDv + w * 128 + g * 4;
#pragma unroll
  for (int nb = 0; nb < 8; ++nb) {
    typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
    const bf16x4 o = {
        static_cast<__bf16>(acc[nb][0] * inv),
        static_cast<__bf16>(acc[nb][1] * inv),
        static_cast<__bf16>(acc[nb][2] * inv),
        static_cast<__bf16>(acc[nb][3] * inv)};
    *reinterpret_cast<bf16x4*>(op + nb * 16) = o;
  }
  if (p.store_lse && w == 0 && g == 0)
    p.lse[static_cast<int64_t>(t) * p.H + n] = l_tot > 0.0f ? (m_i + log2f(l_tot)) * 0.6931471805599453f : -INFINITY;
}

}  // namespace sglang::sparse_mla_prefill_mxfp4_gfx950

namespace sglang {

struct SparseMlaPrefillMxfp4Gfx950Kernel {
  static void run(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView lse,
      double sm_scale,
      int64_t store_lse) {
    using namespace host;
    namespace k = sparse_mla_prefill_mxfp4_gfx950;
    const auto p = make_params(q_nope, q_rope, kv, indices, out, lse, sm_scale, store_lse);
    const int64_t T = q_nope.size(0);
    if (T == 0) return;
    LaunchKernel(static_cast<uint32_t>(T), k::kThreads, q_nope.device())(
        k::sparse_mla_prefill_mxfp4_kernel<false>, p, static_cast<uint32_t>(kv.stride(0)));
  }

  static void run_split(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView lse,
      tvm::ffi::TensorView part_acc,
      tvm::ffi::TensorView part_ml,
      double sm_scale,
      int64_t store_lse,
      int64_t splits) {
    using namespace host;
    namespace k = sparse_mla_prefill_mxfp4_gfx950;
    auto p = make_params(q_nope, q_rope, kv, indices, out, lse, sm_scale, store_lse);
    const int64_t T = q_nope.size(0);
    RuntimeCheck(splits >= 1 && splits <= 65535, "splits must fit grid.y");
    const int64_t parts = T * splits * p.H;
    RuntimeCheck(part_acc.numel() >= parts * k::kDv && part_ml.numel() >= 2 * parts, "workspace");
    if (T == 0) return;
    const int ntiles = (p.topk + k::kKeys - 1) / k::kKeys;
    p.tiles_per_split = (ntiles + static_cast<int>(splits) - 1) / static_cast<int>(splits);
    p.part_acc = static_cast<float*>(part_acc.data_ptr());
    p.part_m = static_cast<float*>(part_ml.data_ptr());
    p.part_l = p.part_m + parts;
    const DLDevice dev = q_nope.device();
    LaunchKernel(dim3(static_cast<uint32_t>(T), static_cast<uint32_t>(splits)), k::kThreads, dev)(
        k::sparse_mla_prefill_mxfp4_kernel<true>, p, static_cast<uint32_t>(kv.stride(0)));
    LaunchKernel(static_cast<uint32_t>(T * p.H), 128, dev)(
        sparse_mla_prefill_gfx950::sparse_mla_split_combine_kernel, p, static_cast<int>(splits));
  }

 private:
  static sparse_mla_prefill_gfx950::Params make_params(
      tvm::ffi::TensorView q_nope,
      tvm::ffi::TensorView q_rope,
      tvm::ffi::TensorView kv,
      tvm::ffi::TensorView indices,
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView lse,
      double sm_scale,
      int64_t store_lse) {
    using namespace host;
    namespace k = sparse_mla_prefill_mxfp4_gfx950;
    constexpr int64_t max_byte = 0x7fffffffLL;
    const int64_t T = q_nope.size(0);
    const int64_t H = q_nope.size(1);
    RuntimeCheck(H >= 1 && H <= 16, "1..16 heads");
    RuntimeCheck(T <= max_byte / H, "query grid too large");
    RuntimeCheck(q_nope.ndim() == 3 && q_nope.size(2) == 512 && q_nope.stride(2) == 1, "q_nope [T, H, 512]");
    RuntimeCheck(
        q_rope.ndim() == 3 && q_rope.size(0) == T && q_rope.size(1) == H && q_rope.size(2) == 64 &&
            q_rope.stride(2) == 1,
        "q_rope [T, H, 64]");
    RuntimeCheck(
        q_nope.stride(0) % 8 == 0 && q_nope.stride(1) % 8 == 0 && q_rope.stride(0) % 8 == 0 &&
            q_rope.stride(1) % 8 == 0 && reinterpret_cast<uintptr_t>(q_nope.data_ptr()) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(q_rope.data_ptr()) % 16 == 0,
        "query strides and base must keep 16-B alignment");
    RuntimeCheck(
        kv.ndim() == 3 && kv.size(1) == 1 && kv.size(2) == k::kRow && kv.stride(2) == 1 &&
            is_type<uint8_t>(kv.dtype()),
        "kv must be uint8 [N, 1, 400] with unit byte columns");
    const int64_t stride = kv.stride(0);
    RuntimeCheck(stride >= k::kRow && stride % 16 == 0, "KV byte stride must be >= 400 and 16-B aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(kv.data_ptr()) % 16 == 0, "KV base must be 16-B aligned");
    RuntimeCheck(stride <= max_byte && kv.size(0) <= max_byte / stride, "KV pool must fit a 31-bit buffer offset");
    RuntimeCheck(
        indices.ndim() == 2 && indices.size(0) == T && indices.stride(1) == 1 && indices.stride(0) == indices.size(1),
        "indices [T, topk] dense");
    // Leave room for the clamped future-tile index prefetch.
    RuntimeCheck(indices.size(1) >= 1 && indices.size(1) <= max_byte - 2 * k::kStages * k::kKeys, "invalid topk");
    RuntimeCheck(out.stride(1) == 512 && out.stride(0) == H * 512 && lse.stride(0) == H, "out / lse dense");
    k::Params p{};
    p.q_nope = static_cast<const __bf16*>(q_nope.data_ptr());
    p.q_rope = static_cast<const __bf16*>(q_rope.data_ptr());
    p.kv = static_cast<const uint8_t*>(kv.data_ptr());
    p.idx = static_cast<const int32_t*>(indices.data_ptr());
    p.kv_bytes = static_cast<uint32_t>(kv.size(0) * stride);
    p.out = static_cast<__bf16*>(out.data_ptr());
    p.lse = static_cast<float*>(lse.data_ptr());
    p.sqn_t = q_nope.stride(0);
    p.sqn_h = q_nope.stride(1);
    p.sqr_t = q_rope.stride(0);
    p.sqr_h = q_rope.stride(1);
    p.H = static_cast<int>(H);
    p.topk = static_cast<int>(indices.size(1));
    p.qk_scale = static_cast<float>(sm_scale * 1.4426950408889634);
    p.store_lse = static_cast<int>(store_lse);
    return p;
  }
};

}  // namespace sglang
