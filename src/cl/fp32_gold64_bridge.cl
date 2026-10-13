// Aevum FP32/GF31 -> natural-order Gold64 coefficient bridge.
#include "base.cl"
#include "fftwidth.cl"
#include "weight.cl"
#include "middle.cl"

#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 8 || NW != 8 || NH != 8
#error Gold64 companion bridge currently requires 512:8:512 geometry
#endif
#if WordSize != 8
#error Gold64 companion bridge requires 64-bit input digits
#endif

#if FFT_TYPE == FFT32 && FFT_FP32 && !NTT_GF31 && !NTT_GF61

// 16x16 import uses 4352 bytes of LDS instead of the stock 64x64 long tile.
// Read and write whole signed pairs so every input bit reaches both planes.
KERNEL(256) gold64_fp32_import(P(Word2) out, CP(Word2) in) {
  local Word2 tile[16][17];
  const u32 lid = get_local_id(0), group = get_group_id(0);
  const u32 tx = lid & 15, ty = lid >> 4;
  const u32 rowBase = (group % (BIG_HEIGHT/16)) * 16;
  const u32 colBase = (group / (BIG_HEIGHT/16)) * 16;
  tile[ty][tx] = in[(colBase + ty) * BIG_HEIGHT + rowBase + tx];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[(rowBase + ty) * WIDTH + colBase + tx] = tile[tx][ty];
}

// Input pair p = row*WIDTH+column represents natural digits
// j=2*(column*BIG_HEIGHT+row), j+1. fftW leaves the components swapped.
// fftMiddleOut already applies 1/(2*NWORDS); inverse weights undo the
// two input factors 2^-24. Do NOT divide by N or N/2 again here.
KERNEL(256) gold64_fp32_unweight(CP(F2) in, P(F2) out, BigTabFP32 THREAD_WEIGHTS) {
  local F2 tile[16][17];
  const u32 lid = get_local_id(0), group = get_group_id(0);
  const u32 tx = lid & 15, ty = lid >> 4;
  const u32 colBase = (group % (WIDTH/16)) * 16;
  const u32 rowBase = (group / (WIDTH/16)) * 16;
  const u32 col = colBase + tx, row = rowBase + ty;
  const u32 me = col % G_W, gx = col / G_W;
  const u32 word = 2 * (col * BIG_HEIGHT + row);
  const u32 frac = fracBits(word) - 1u;
  const u32 stepFrac = weightStepFracBits(gx);
  const u32 baseFrac = fracBits(me * BIG_HEIGHT * 2) + stepFrac;
  F base = fancyMul(THREAD_WEIGHTS[me].x, iweightStep(gx));
  base = optionalDouble(base, baseFrac > stepFrac);
  const F w0 = optionalDouble(fancyMul(base, THREAD_WEIGHTS[G_W + row].x), frac > baseFrac);
  const F w1 = optionalDouble(fancyMul(w0, IWEIGHT_STEP), frac + FRAC_BPW_HI > FRAC_BPW_HI);
  const F2 x = in[row * WIDTH + col];
  tile[ty][tx] = (F2)(x.y * w0, x.x * w1);
  barrier(CLK_LOCAL_MEM_FENCE);
  out[(colBase + ty) * BIG_HEIGHT + rowBase + tx] = tile[tx][ty];
}

#elif FFT_TYPE == FFT31 && !FFT_FP32 && NTT_GF31 && !NTT_GF61

KERNEL(256) gold64_m31_import(P(Word2) out, CP(Word2) in) {
  local Word2 tile[16][17];
  const u32 lid = get_local_id(0), group = get_group_id(0);
  const u32 tx = lid & 15, ty = lid >> 4;
  const u32 rowBase = (group % (BIG_HEIGHT/16)) * 16;
  const u32 colBase = (group / (BIG_HEIGHT/16)) * 16;
  tile[ty][tx] = in[(colBase + ty) * BIG_HEIGHT + rowBase + tx];
  barrier(CLK_LOCAL_MEM_FENCE);
  out[(rowBase + ty) * WIDTH + colBase + tx] = tile[tx][ty];
}

// Fuse the final GF31 width transform with inverse weighting and natural-order
// residue export. This removes the transform-sized intermediate written by
// fftWGF31 and immediately reread by gold64_m31_unweight.
KERNEL(G_W) gold64_m31_fftW_unweight(P(u32) out, CP(T2) in, Trig smallTrig) {
  local GF31 lds[LDS_BYTES / sizeof(GF31)];
  CP(GF31) in31 = (CP(GF31)) (in + DISTGF31);
  TrigGF31 smallTrig31 = (TrigGF31) (smallTrig + DISTWTRIGGF31);

  GF31 u[NW];
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);

  dependentLaunchWait();
  readCarryFusedLine(in31, u, g, me);
  fft_WIDTH(lds, u, smallTrig31, 1, me);

  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 =
      (bigword_weight_shift + 30) % 31;

  union { uint2 a; u64 b; } combo;
  const u64 combo_step =
      make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  const u64 combo_bigstep =
      (comboFracBits(G_W * BIG_HEIGHT * 2 - 1) +
       make_u64((G_W * BIG_HEIGHT * 2 - 1) *
                    bigword_weight_shift_minus1,
                0)) % (31ULL << 32);

  u32 word_index = (me * BIG_HEIGHT + g) * 2;
  combo.b =
      comboFracBits(word_index) +
      make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  const u32 log2_NWORDS =
      (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
      (PFA_RADIX ? 0 :
       (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 :
        MIDDLE == 8 ? 3 : 4)) +
      (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 :
       SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  combo.a[1] = (combo.a[1] + log2_NWORDS + 1) % 31;
  const u32 integer_bpw_shift = (log2_NWORDS + 1) % 31;

#pragma unroll
  for (u32 i = 0; i < NW; ++i) {
    const u32 shift0 = (EXP % NWORDS == 0) ? integer_bpw_shift : combo.a[1];
    combo.b += combo_step;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
    const u32 shift1 = (EXP % NWORDS == 0) ? integer_bpw_shift : combo.a[1];

    const GF31 v = SWAP_XY(u[i]);
    out[word_index] = get_Z31(shr(v.x, shift0));
    out[word_index + 1u] = get_Z31(shr(v.y, shift1));

    combo.b += combo_bigstep;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
    word_index += G_W * BIG_HEIGHT * 2u;
  }
}

// Export the standalone GF31 fftW state as natural-order coefficient residues.
// This is the same inverse NTT weight logic as debugPrecarry3161M31.
KERNEL(G_W) gold64_m31_unweight(CP(GF31) in, P(u32) out) {
  const u32 g = get_group_id(0);
  const u32 me = get_local_id(0);
  const u32 gx = g % NW;
  const u32 gy = g / NW;
  const u32 H = BIG_HEIGHT;
  const u32 line = gy * CARRY_LEN;
  u32 word_index = (gx * G_W * H + me * H + line) * 2;

  const u32 log2_root_two = M31_LOG2_ROOT_TWO;
  const u32 bigword_weight_shift =
      (NWORDS - EXP % NWORDS) * log2_root_two % 31;
  const u32 bigword_weight_shift_minus1 =
      (bigword_weight_shift + 30) % 31;

  union { uint2 a; u64 b; } combo;
  const u64 combo_step =
      make_u64(bigword_weight_shift_minus1, FRAC_BPW_HI);
  combo.b =
      comboFracBits(word_index) +
      make_u64(word_index * bigword_weight_shift_minus1, 0xFFFFFFFF);

  const u32 log2_NWORDS =
      (WIDTH == 256 ? 8 : WIDTH == 512 ? 9 : WIDTH == 1024 ? 10 : 12) +
      (PFA_RADIX ? 0 :
       (MIDDLE == 1 ? 0 : MIDDLE == 2 ? 1 : MIDDLE == 4 ? 2 :
        MIDDLE == 8 ? 3 : 4)) +
      (SMALL_HEIGHT == 256 ? 8 : SMALL_HEIGHT == 512 ? 9 :
       SMALL_HEIGHT == 1024 ? 10 : 12) + 1;
  combo.a[1] = (combo.a[1] + log2_NWORDS + 1) % 31;
  const u32 integer_bpw_shift = (log2_NWORDS + 1) % 31;

  for (i32 i = 0; i < CARRY_LEN; ++i) {
    const u32 p = G_W * gx + WIDTH * (CARRY_LEN * gy + i) + me;
    const u32 shift0 = (EXP % NWORDS == 0) ? integer_bpw_shift : combo.a[1];
    combo.b += combo_step;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
    const u32 shift1 = (EXP % NWORDS == 0) ? integer_bpw_shift : combo.a[1];

    const GF31 u = SWAP_XY(in[p]);
    const u32 natural = word_index + 2u * i;
    out[natural] = get_Z31(shr(u.x, shift0));
    out[natural + 1u] = get_Z31(shr(u.y, shift1));

    combo.b += combo_step;
    combo.a[1] = adjust_m31_weight_shift(combo.a[1]);
  }
}

#else
#error Gold64 companion bridge requires either FP32-only or GF31-only fixed transform
#endif
