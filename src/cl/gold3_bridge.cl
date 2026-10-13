// Gold3 wide digits -> optimized Aevum quadratic-extension transforms.
#include "base.cl"
#include "fftwidth.cl"
#include "middle.cl"
#include "gold3_arith.cl"
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 4 || NW != 8 || NH != 8 || WordSize != 8
#error Gold3 requires 512:4:512 with 64-bit reduced modular input
#endif
#if FFT_TYPE == FFT61 && NTT_GF61 && !NTT_GF31 && !FFT_FP32 && !FFT_FP64
#define G3_GF GF61
#define G3_TRIG TrigGF61
#define G3_SCALAR u64
#define G3_PERIOD 61u
#define G3_ROOT_TWO M61_LOG2_ROOT_TWO
#define G3_REDUCE g3_digit61
#define G3_MAKE make_Z61
#define G3_GET get_Z61
#elif FFT_TYPE == FFT31 && NTT_GF31 && !NTT_GF61 && !FFT_FP32 && !FFT_FP64
#define G3_GF GF31
#define G3_TRIG TrigGF31
#define G3_SCALAR u32
#define G3_PERIOD 31u
#define G3_ROOT_TWO M31_LOG2_ROOT_TWO
#define G3_REDUCE g3_digit31
#define G3_MAKE make_Z31
#define G3_GET get_Z31
#else
#error Gold3 companion requires exactly one modular plane
#endif

// An exact weight exponent: e_j = (-q*j) mod N and rho=2^(N^-1 mod period).
// Unsigned multiplication wrap is safe since N=2^21 divides 2^32. No rounded
// fractional counter, special digit-zero adjustment or per-digit division.
static inline u32 gold3_weight_shift(u32 j) {
  return (((0u - (u32)EXP * j) & (NWORDS - 1u)) * G3_ROOT_TWO) % G3_PERIOD;
}

KERNEL(256) gold3_import(P(Word2) out, __global const g3_digit* in) {
  local Word2 tile[16][17];
  const u32 lid = get_local_id(0), group = get_group_id(0);
  const u32 tx = lid & 15, ty = lid >> 4;
  const u32 rowBase = (group % (BIG_HEIGHT/16)) * 16;
  const u32 colBase = (group / (BIG_HEIGHT/16)) * 16;
  const u32 j = 2 * ((colBase + ty) * BIG_HEIGHT + rowBase + tx);
  tile[ty][tx] = (Word2)(G3_REDUCE(in[j]), G3_REDUCE(in[j+1]));
  barrier(CLK_LOCAL_MEM_FENCE);
  out[(rowBase + ty) * WIDTH + colBase + tx] = tile[tx][ty];
}

KERNEL(G_W) gold3_fftP(P(G3_GF) out, CP(Word2) in, G3_TRIG roots) {
  local G3_GF lds[LDS_BYTES / sizeof(G3_GF)];
  G3_GF u[NW];
  const u32 row = get_group_id(0), me = get_local_id(0);
#pragma unroll
  for (u32 i = 0; i < NW; ++i) {
    const u32 col = me + G_W*i, j = 2*(col*BIG_HEIGHT+row);
    const Word2 x = in[row*WIDTH+col];
    u[i] = U2(shl(G3_MAKE(x.x), gold3_weight_shift(j)),
              shl(G3_MAKE(x.y), gold3_weight_shift(j+1)));
  }
  fft_WIDTH(lds, u, roots, 1, me);
  writeCarryFusedLine(u, out, row, me);
}

// fftMiddleOut/fftW leave swapped components and a factor 2*N. Fuse the
// existing optimized width FFT with exact inverse weights and natural export.
// 2*N=2^22; its inverse is a cyclic field shift, not an integer division.
KERNEL(G_W) gold3_fftW_export(P(G3_SCALAR) out, CP(G3_GF) in, G3_TRIG roots) {
  local G3_GF lds[LDS_BYTES / sizeof(G3_GF)];
  G3_GF u[NW];
  const u32 row = get_group_id(0), me = get_local_id(0);
  readCarryFusedLine(in, u, row, me);
  fft_WIDTH(lds, u, roots, 1, me);
#pragma unroll
  for (u32 i = 0; i < NW; ++i) {
    const u32 j = 2*((me+G_W*i)*BIG_HEIGHT+row);
    out[j] = G3_GET(shr(u[i].y, (gold3_weight_shift(j)+22u) % G3_PERIOD));
    out[j+1] = G3_GET(shr(u[i].x, (gold3_weight_shift(j+1)+22u) % G3_PERIOD));
  }
}
