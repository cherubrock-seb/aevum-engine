// Aevum FP32 -> natural-order Gold64 coefficient bridge.
// Reuses the inverse-weight formulas in carry.cl without rounding or carry.
#include "base.cl"
#include "fftwidth.cl"
#include "weight.cl"

#if FFT_TYPE != FFT32 || !FFT_FP32 || NTT_GF31 || NTT_GF61 || WordSize != 8
#error FP32 Gold64 bridge requires the FP32-only transform with 64-bit input digits
#endif
#if WIDTH != 512 || SMALL_HEIGHT != 512 || MIDDLE != 8 || NW != 8 || NH != 8
#error FP32 Gold64 bridge currently requires 512:8:512 geometry
#endif

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
// A padded tile fuses weight removal and transpose, coalescing both sides.
KERNEL(256) gold64_fp32_unweight(CP(F2) in, P(F2) out, BigTabFP32 THREAD_WEIGHTS) {
  local F2 tile[16][17];
  const u32 lid = get_local_id(0), group = get_group_id(0);
  const u32 tx = lid & 15, ty = lid >> 4;
  const u32 colBase = (group % (WIDTH/16)) * 16;
  const u32 rowBase = (group / (WIDTH/16)) * 16;
  const u32 col = colBase + tx, row = rowBase + ty;
  const u32 me = col % G_W, gx = col / G_W;
  const u32 word = 2 * (col * BIG_HEIGHT + row);
  const u32 frac = fracBits(word);
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
