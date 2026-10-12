// Packed real-length Gold64 IBDWT adapted from Marin, Copyright 2025 Yves Gallot.
// MIT notice: licenses/Marin-MIT.txt. Scope: the 4M-word power-of-two schedule.
// Each ulong2 is two adjacent natural-order digits, NOT a GF(M61^2) value.
// Forward + fused pointwise/inverse + backward returns (N/2)*convolution.
// Input weights and inverse weights use Gold64Plan.h's exact ceil(q*j/N) law.
// No carry, reconstruction, capacity promise or production selection here.
#include "gold64_arith.cl"
#if !defined(GOLD64_N) || GOLD64_N != 4194304
#error Gold64 transform currently requires exactly 4194304 real words
#endif
#define G64_LN_SZ_S5 20
#define G64_CHUNK64 8u
#define G64_ROOT_EXTRA_ARGS
#define G64_ROOT_R2_DECL __global const ulong * restrict const r2 = root
#define G64_ROOT_R2I_DECL __global const ulong * restrict const r2i = root + GOLD64_N/2
#define G64_ROOT_R2_2_DECL __global const ulong2 * restrict const r2_2 = (__global const ulong2 *)root
#define G64_ROOT_R2I_2_DECL __global const ulong2 * restrict const r2i_2 = (__global const ulong2 *)(root + GOLD64_N/2)
#define G64_ROOT_R4_DECL __global const ulong2 * restrict const r4 = (__global const ulong2 *)(root + GOLD64_N)
#define G64_ROOT_R4I_DECL __global const ulong2 * restrict const r4i = (__global const ulong2 *)(root + 2*GOLD64_N)


static inline void gold64_fwd4(ulong2 * const x, const ulong r1, const ulong2 r23)
{
	const ulong2 u0 = x[0], u2 = gold64_mul2(x[2], r1), u1 = gold64_mul2(x[1], r23.s0), u3 = gold64_mul2(x[3], r23.s1);
	const ulong2 v0 = gold64_add2(u0, u2), v2 = gold64_sub2(u0, u2), v1 = gold64_add2(u1, u3), v3 = gold64_muli2(gold64_sub2(u1, u3));
	x[0] = gold64_add2(v0, v1); x[1] = gold64_sub2(v0, v1); x[2] = gold64_add2(v2, v3); x[3] = gold64_sub2(v2, v3);
}

static inline void gold64_bck4(ulong2 * const x, const ulong r1i, const ulong2 r23i)
{
	const ulong2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];
	const ulong2 v0 = gold64_add2(u0, u1), v1 = gold64_sub2(u0, u1), v2 = gold64_add2(u3, u2), v3 = gold64_muli2(gold64_sub2(u3, u2));
	x[0] = gold64_add2(v0, v2); x[2] = gold64_mul2(gold64_sub2(v0, v2), r1i); x[1] = gold64_mul2(gold64_add2(v1, v3), r23i.s0); x[3] = gold64_mul2(gold64_sub2(v1, v3), r23i.s1);
}

static inline void gold64_fwd4_0(ulong2 * const x)
{
	const ulong2 u0 = x[0], u2 = x[2], u1 = x[1], u3 = x[3];
	const ulong2 v0 = gold64_add2(u0, u2), v2 = gold64_sub2(u0, u2), v1 = gold64_add2(u1, u3), v3 = gold64_muli2(gold64_sub2(u1, u3));
	x[0] = gold64_add2(v0, v1); x[1] = gold64_sub2(v0, v1); x[2] = gold64_add2(v2, v3); x[3] = gold64_sub2(v2, v3);
}

static inline void gold64_bck4_0(ulong2 * const x)
{
	const ulong2 u0 = x[0], u1 = x[1], u2 = x[2], u3 = x[3];
	const ulong2 v0 = gold64_add2(u0, u1), v1 = gold64_sub2(u0, u1), v2 = gold64_add2(u3, u2), v3 = gold64_muli2(gold64_sub2(u3, u2));
	x[0] = gold64_add2(v0, v2); x[2] = gold64_sub2(v0, v2); x[1] = gold64_add2(v1, v3); x[3] = gold64_sub2(v1, v3);
}

static inline void gold64_fwd22(ulong2 * const x, const ulong2 r)
{
	const ulong2 u0 = x[0], u1 = gold64_mul2(x[1], r);
	x[0] = gold64_add2(u0, u1); x[1] = gold64_sub2(u0, u1);
}

static inline void gold64_bck22(ulong2 * const x, const ulong2 ri)
{
	const ulong2 u0 = x[0], u1 = x[1];
	x[0] = gold64_add2(u0, u1); x[1] = gold64_mul2(gold64_sub2(u0, u1), ri);
}

static inline void gold64_sqr22(ulong2 * const x, const ulong r)
{
	const ulong2 sx0 = gold64_sqr2(x[0]), sx1 = gold64_sqr2(x[1]);
	x[0].s1 = gold64_mul(x[0].s1, gold64_add(x[0].s0, x[0].s0)); x[0].s0 = gold64_add(sx0.s0, gold64_mul(sx0.s1, r));
	x[1].s1 = gold64_mul(x[1].s1, gold64_add(x[1].s0, x[1].s0)); x[1].s0 = gold64_sub(sx1.s0, gold64_mul(sx1.s1, r));
}

static inline void gold64_mul22(ulong2 * const x, const ulong2 * const y, const ulong r)
{
	const ulong2 xy0 = gold64_mul2(x[0], y[0]), xy1 = gold64_mul2(x[1], y[1]);
	x[0].s1 = gold64_add(gold64_mul(x[0].s0, y[0].s1), gold64_mul(x[0].s1, y[0].s0)); x[0].s0 = gold64_add(xy0.s0, gold64_mul(xy0.s1, r));
	x[1].s1 = gold64_add(gold64_mul(x[1].s0, y[1].s1), gold64_mul(x[1].s1, y[1].s0)); x[1].s0 = gold64_sub(xy1.s0, gold64_mul(xy1.s1, r));
}

static inline void gold64_sqr_4(ulong2 * const xl, const ulong2 r, const ulong2 ri)
{
	gold64_fwd22(&xl[0], r.s0); gold64_fwd22(&xl[2], r.s1);
	gold64_sqr22(&xl[0], r.s0); gold64_sqr22(&xl[2], r.s1);
	gold64_bck22(&xl[0], ri.s0); gold64_bck22(&xl[2], ri.s1);
}

static inline void gold64_mul_4(ulong2 * const xl, const ulong2 * const yl, const ulong2 r, const ulong2 ri)
{
	gold64_fwd22(&xl[0], r.s0); gold64_fwd22(&xl[2], r.s1);
	gold64_mul22(&xl[0], &yl[0], r.s0); gold64_mul22(&xl[2], &yl[2], r.s1);
	gold64_bck22(&xl[0], ri.s0); gold64_bck22(&xl[2], ri.s1);
}

static inline void gold64_loadg2(const uint n, ulong2 * const xl, __global const ulong2 * restrict const x, const uint s) { for (uint l = 0; l < n; ++l) xl[l] = x[l * s]; }

static inline void gold64_loadl2(const uint n, ulong2 * const xl, __local const ulong2 * restrict const X, const uint s) { for (uint l = 0; l < n; ++l) xl[l] = X[l * s]; }

static inline void gold64_storeg2(const uint n, __global ulong2 * restrict const x, const uint s, const ulong2 * const xl) { for (uint l = 0; l < n; ++l) x[l * s] = xl[l]; }

static inline void gold64_storel2(const uint n, __local ulong2 * restrict const X, const uint s, const ulong2 * const xl) { for (uint l = 0; l < n; ++l) X[l * s] = xl[l]; }

static inline void gold64_forward_4(const uint m, __local ulong2 * restrict const X, const ulong r1, const ulong2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, m);
	gold64_fwd4(xl, r1, r23);
	gold64_storel2(4, X, m, xl);
}

static inline void gold64_forward_4i(const uint ml, __local ulong2 * restrict const X,
	const uint mg, __global const ulong2 * restrict const x, const ulong r1, const ulong2 r23)
{
	ulong2 xl[4]; gold64_loadg2(4, xl, x, mg);
	gold64_fwd4(xl, r1, r23);
	gold64_storel2(4, X, ml, xl);
}

static inline void gold64_forward_4o(const uint mg, __global ulong2 * restrict const x,
	const uint ml, __local const ulong2 * restrict const X, const ulong r1, const ulong2 r23)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, ml);
	gold64_fwd4(xl, r1, r23);
	gold64_storeg2(4, x, mg, xl);
}

static inline void gold64_backward_4(const uint m, __local ulong2 * restrict const X, const ulong r1i, const ulong2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, m);
	gold64_bck4(xl, r1i, r23i);
	gold64_storel2(4, X, m, xl);
}

static inline void gold64_backward_4i(const uint ml, __local ulong2 * restrict const X,
	const uint mg, __global const ulong2 * restrict const x, const ulong r1i, const ulong2 r23i)
{
	ulong2 xl[4]; gold64_loadg2(4, xl, x, mg);
	gold64_bck4(xl, r1i, r23i);
	gold64_storel2(4, X, ml, xl);
}

static inline void gold64_backward_4o(const uint mg, __global ulong2 * restrict const x,
	const uint ml, __local const ulong2 * restrict const X, const ulong r1i, const ulong2 r23i)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, ml);
	gold64_bck4(xl, r1i, r23i);
	gold64_storeg2(4, x, mg, xl);
}

static inline void gold64_forward_4i_0(const uint ml, __local ulong2 * restrict const X,
	const uint mg, __global const ulong2 * restrict const x)
{
	ulong2 xl[4]; gold64_loadg2(4, xl, x, mg);
	gold64_fwd4_0(xl);
	gold64_storel2(4, X, ml, xl);
}

static inline void gold64_backward_4o_0(const uint mg, __global ulong2 * restrict const x,
	const uint ml, __local const ulong2 * restrict const X)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, ml);
	gold64_bck4_0(xl);
	gold64_storeg2(4, x, mg, xl);
}

static inline void gold64_forward_mul_4o(__global ulong2 * restrict const x, __local const ulong2 * restrict const X, const ulong2 r)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, 1);
	gold64_fwd22(&xl[0], r.s0); gold64_fwd22(&xl[2], r.s1);
	gold64_storeg2(4, x, 1, xl);
}

static inline void gold64_square_4(__local ulong2 * restrict const X, const ulong2 r, const ulong2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, 1);
	gold64_sqr_4(xl, r, ri);
	gold64_storel2(4, X, 1, xl);
}

static inline void gold64_mult_4(__local ulong2 * restrict const X, __global const ulong2 * restrict const y, const ulong2 r, const ulong2 ri)
{
	barrier(CLK_LOCAL_MEM_FENCE);
	ulong2 xl[4]; gold64_loadl2(4, xl, X, 1);
	ulong2 yl[4]; gold64_loadg2(4, yl, y, 1);
	gold64_mul_4(xl, yl, r, ri);
	gold64_storel2(4, X, 1, xl);
}

#define G64_DECLARE_VAR_REG() \
	__global ulong2 * restrict const x = (__global ulong2 *)(&reg[offset]); \
	G64_ROOT_R2_DECL; \
	G64_ROOT_R2I_DECL; \
	G64_ROOT_R2_2_DECL; \
	G64_ROOT_R2I_2_DECL; \
	G64_ROOT_R4_DECL; \
	G64_ROOT_R4I_DECL; \
	const uint id = (uint)get_global_id(0);

#define G64_DECLARE_VAR(B_N, CHUNK_N) \
	G64_DECLARE_VAR_REG(); \
	\
	__local ulong2 X[4 * B_N * CHUNK_N]; \
	\
	/* thread_idx < B_N */ \
	const uint local_id = id % (B_N * CHUNK_N), group_id = id / (B_N * CHUNK_N); \
	const uint i = local_id, chunk_idx = i % CHUNK_N, thread_idx = i / CHUNK_N, block_idx = group_id * CHUNK_N + chunk_idx; \
	__local ulong2 * const Xi = &X[chunk_idx]; \
	\
	const uint block_idx_m = block_idx >> lm, idx_m = block_idx_m * B_N + thread_idx; \
	const uint block_idx_mm = block_idx_m << lm, idx_mm = idx_m << lm; \
	\
	const uint ki = block_idx + block_idx_mm * (B_N * 3 - 1) + idx_mm, ko = block_idx - block_idx_mm + idx_mm * 4; \
	const uint sj = s + idx_m;

#define G64_FORWARD_64_80(CHUNK_N) \
	const uint i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	gold64_forward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2[sj / 4], r4[sj / 4]); \
	gold64_forward_4o(1u << lm, &x[ko], 1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], r2[sj], r4[sj]);

#define G64_BACKWARD_64_80(CHUNK_N) \
	gold64_backward_4i(1 * CHUNK_N, &Xi[CHUNK_N * 4 * thread_idx], 1u << lm, &x[ko], r2i[sj], r4i[sj]); \
	const uint i4 = 4 * (thread_idx & ~(4 - 1)) + (thread_idx % 4); \
	gold64_backward_4(4 * CHUNK_N, &Xi[CHUNK_N * i4], r2i[sj / 4], r4i[sj / 4]);

#define G64_DECLARE_VAR_1024() \
	__local ulong2 X[1024]; \
	\
	G64_DECLARE_VAR_REG(); \
	const uint j = id, sj = GOLD64_N / 8 + j, k = 4 * id, i = k % 1024; \
	const uint sj2 = sj / 2, k2 = 4 * (id & ~(2 - 1)) + (id % 2), i2 = k2 % 1024; \
	const uint sj8 = sj / 8, k8 = 4 * (id & ~(8 - 1)) + (id % 8), i8 = k8 % 1024; \
	const uint sj32 = sj / 32, k32 = 4 * (id & ~(32 - 1)) + (id % 32), i32 = k32 % 1024; \
	const uint sj128 = sj / 128, k128 = 4 * (id & ~(128 - 1)) + (id % 128), i128 = k128 % 1024;

__kernel
__attribute__((reqd_work_group_size(128,1,1)))
void gold64_forward64(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset,
	const uint s, const uint lm)
{
	G64_DECLARE_VAR(64 / 4, G64_CHUNK64);

	gold64_forward_4i(16 * G64_CHUNK64, &X[i], 16u << lm, &x[ki], r2[sj / 16], r4[sj / 16]);
	G64_FORWARD_64_80(G64_CHUNK64);
}

__kernel
__attribute__((reqd_work_group_size(128,1,1)))
void gold64_backward64(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset,
	const uint s, const uint lm)
{
	G64_DECLARE_VAR(64 / 4, G64_CHUNK64);

	G64_BACKWARD_64_80(G64_CHUNK64);
	gold64_backward_4o(16u << lm, &x[ki], 16 * G64_CHUNK64, &X[i], r2i[sj / 16], r4i[sj / 16]);
}

__kernel
__attribute__((reqd_work_group_size(128,1,1)))
void gold64_forward64_0(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset)
{
	const uint s = 64 / 4; const uint lm = G64_LN_SZ_S5 - 1 - 4;
	G64_DECLARE_VAR(64 / 4, G64_CHUNK64);

	gold64_forward_4i_0(16 * G64_CHUNK64, &X[i], 16u << lm, &x[ki]);
	G64_FORWARD_64_80(G64_CHUNK64);
}

__kernel
__attribute__((reqd_work_group_size(128,1,1)))
void gold64_backward64_0(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset)
{
	const uint s = 64 / 4; const uint lm = G64_LN_SZ_S5 - 1 - 4;
	G64_DECLARE_VAR(64 / 4, G64_CHUNK64);

	G64_BACKWARD_64_80(G64_CHUNK64);
	gold64_backward_4o_0(16u << lm, &x[ki], 16 * G64_CHUNK64, &X[i]);
}

__kernel
__attribute__((reqd_work_group_size(256,1,1)))
void gold64_forward_mul1024(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset)
{
	G64_DECLARE_VAR_1024();
	gold64_forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	gold64_forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	gold64_forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	gold64_forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	gold64_forward_mul_4o(&x[k], &X[i], r2_2[sj]);
}

__kernel
__attribute__((reqd_work_group_size(256,1,1)))
void gold64_sqr1024(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset)
{
	G64_DECLARE_VAR_1024();

	gold64_forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	gold64_forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	gold64_forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	gold64_forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	gold64_square_4(&X[i], r2_2[sj], r2i_2[sj]);
	gold64_backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	gold64_backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	gold64_backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);
	gold64_backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);
}

__kernel
__attribute__((reqd_work_group_size(256,1,1)))
void gold64_mul1024(__global ulong * restrict const reg, __global const ulong * restrict const root G64_ROOT_EXTRA_ARGS, const ulong offset, __global const ulong * restrict const prepared, const ulong offset_y)
{
	G64_DECLARE_VAR_1024();
	__global const ulong2 * restrict const y = (__global const ulong2 *)(&prepared[offset_y]);

	gold64_forward_4i(128, &X[i128], 128, &x[k128], r2[sj128], r4[sj128]);
	gold64_forward_4(32, &X[i32], r2[sj32], r4[sj32]);
	gold64_forward_4(8, &X[i8], r2[sj8], r4[sj8]);
	gold64_forward_4(2, &X[i2], r2[sj2], r4[sj2]);
	gold64_mult_4(&X[i], &y[k], r2_2[sj], r2i_2[sj]);
	gold64_backward_4(2, &X[i2], r2i[sj2], r4i[sj2]);
	gold64_backward_4(8, &X[i8], r2i[sj8], r4i[sj8]);
	gold64_backward_4(32, &X[i32], r2i[sj32], r4i[sj32]);
	gold64_backward_4o(128, &x[k128], 128, &X[i128], r2i[sj128], r4i[sj128]);
}

// Import signed natural-order digits; arithmetic is exact modulo p for every
// signed 64-bit input. Hybrid reconstruction still needs a separate bound.
static inline ulong gold64_from_signed(long x) {
  const ulong bits = as_ulong(x);
  return x < 0 ? GOLD64_P - (~bits + 1UL) : bits;
}
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_weight(__global const long4 * restrict digits,
                   __global ulong4 * restrict reg,
                   __global const ulong2 * restrict weights) {
  const uint k = (uint)get_global_id(0);
  const long4 d = digits[k];
  reg[k] = (ulong4)(
    gold64_mul(gold64_from_signed(d.s0), weights[k].x),
    gold64_mul(gold64_from_signed(d.s1), weights[k + GOLD64_N/4].x),
    gold64_mul(gold64_from_signed(d.s2), weights[k + GOLD64_N/2].x),
    gold64_mul(gold64_from_signed(d.s3), weights[k + 3*(GOLD64_N/4)].x));
}
// weights[].y is inverse_weight/(N/2), precombined once on the host.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold64_unweight(__global const ulong4 * restrict reg,
                     __global ulong4 * restrict residues,
                     __global const ulong2 * restrict weights) {
  const uint k = (uint)get_global_id(0);
  const ulong4 x = reg[k];
  residues[k] = (ulong4)(
    gold64_mul(x.s0, weights[k].y),
    gold64_mul(x.s1, weights[k + GOLD64_N/4].y),
    gold64_mul(x.s2, weights[k + GOLD64_N/2].y),
    gold64_mul(x.s3, weights[k + 3*(GOLD64_N/4)].y));
}
