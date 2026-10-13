// Exact CRT and O(N) segmented cyclic carry. No FP, division or CPU roundtrip.
#include "gold3_arith.cl"
#define G3_SEGMENT 256u
#define G3_SEGMENTS (G3_WORDS / G3_SEGMENT)
#define G3_SCAN 256u
#define G3_GROUPS (G3_SEGMENTS / G3_SCAN)
#define G3_BAD_RESIDUE 1u
#define G3_BAD_RANGE 2u
#define G3_OVERFLOW 4u
#define G3_BAD_CARRY 8u
#define G3_NO_FIXED_POINT 16u
#define G3_EXIT_MISMATCH 32u

static inline void g3_error(__global uint* status, uint flag, uint segment) {
  atomic_or((volatile __global uint*)status, flag);
  atomic_min((volatile __global uint*)(status+1), segment);
  atomic_inc((volatile __global uint*)(status+2));
}
static inline g3_i192 g3_checked_add(g3_i192 a, g3_i192 b,
                                    __global uint* status, uint segment) {
  const g3_i192 r = g3_add(a, b);
  if (g3_add_overflow(a, b, r)) { g3_error(status, G3_OVERFLOW, segment); }
  return r;
}
static inline g3_i192 g3_state(uint index) {
  return index == 0 ? g3_make(~0UL, ~0UL, ~0UL) : g3_make(index == 2, 0, 0);
}
static inline uint g3_state_index(g3_i192 x, __global uint* status, uint segment) {
  if (g3_equal(x, g3_zero())) { return 1; }
  if (g3_equal(x, g3_state(0))) { return 0; }
  if (g3_equal(x, g3_state(2))) { return 2; }
  g3_error(status, G3_BAD_CARRY, segment);
  return 1;
}
static inline uint g3_eval(uint4 map, uint state) {
  return state == 0 ? map.x : (state == 1 ? map.y : map.z);
}
// First a, then b. Composition is associative and order must be preserved.
static inline uint4 g3_compose(uint4 a, uint4 b) {
  return (uint4)(g3_eval(b, a.x), g3_eval(b, a.y), g3_eval(b, a.z), 0);
}
static inline uint4 g3_identity(void) { return (uint4)(0, 1, 2, 0); }

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_crt(__global const ulong* gold, __global const ulong* m61,
              __global const uint* m31, __global g3_i192* coefficients,
              __global uint* status, uint factor, uint subtractTwo) {
  const uint j = get_global_id(0), segment = j / G3_SEGMENT;
  g3_i192 x = g3_zero();
  if (gold[j] >= G3_G || m61[j] >= G3_P61 || m31[j] >= G3_P31) {
    g3_error(status, G3_BAD_RESIDUE, segment);
  } else {
    x = g3_crt(gold[j], m61[j], m31[j]);
    if (!g3_coefficient_in_range(x)) { g3_error(status, G3_BAD_RANGE, segment); }
  }
  if (factor == 3) {
    x = g3_checked_add(g3_checked_add(x, x, status, segment), x, status, segment);
  }
  if (subtractTwo && j == 0) { x = g3_checked_add(x, g3_make(~1UL, ~0UL, ~0UL), status, segment); }
  coefficients[j] = x;
}

// Linear operations enter the SAME carry machinery. Input digits are bounded
// on upload and normalized output; no 64-bit narrowing of signs or carries.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_linear(__global const g3_digit* lhs, __global const g3_digit* rhs,
                 __global g3_i192* coefficients, __global uint* status,
                 uint operation, uint value) {
  const uint j = get_global_id(0), segment = j / G3_SEGMENT;
  g3_i192 x = g3_from_digit(lhs[j]);
  if (operation == 1) { x = g3_checked_add(x, g3_from_digit(rhs[j]), status, segment); }
  if (operation == 2) { x = g3_checked_add(x, g3_negate(g3_from_digit(rhs[j])), status, segment); }
  if (operation == 3 && j == 0) {
    x = g3_checked_add(x, g3_negate(g3_make(value, 0, 0)), status, segment);
  }
  coefficients[j] = x;
}

// First normalize every 256-digit segment with zero incoming carry. Preserve
// its wide tail T_s; all stored digits become balanced. Since |C|<2^154 and
// width>=64, |T_s|<2^91. Only 256 sequential steps per work-item, 8192 segments.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_segment_zero(__global const g3_i192* coefficients,
                        __global g3_digit* baseDigits, __global g3_i192* tails,
                        __global const uchar* widths, __global uint* status) {
  const uint s = get_global_id(0), first = s * G3_SEGMENT;
  g3_i192 carry = g3_zero();
  for (uint i = 0; i < G3_SEGMENT; ++i) {
    const uint j = first + i;
    g3_digit digit;
    carry = g3_split(g3_checked_add(coefficients[j], carry, status, s), widths[j], &digit);
    baseDigits[j] = digit;
  }
  const g3_i192 magnitude = g3_abs(carry);
  if (magnitude.z || magnitude.y >= (1UL << 27)) { g3_error(status, G3_BAD_CARRY, s); }
  tails[s] = carry;
}

// Actual tail is T_s + delta_s, delta_s in {-1,0,1}. Derive the three-state
// transfer map by injecting T_(s-1)+delta_(s-1) into balanced base digits.
// Each segment radix exceeds 2^16384, while incoming magnitude is <2^91;
// therefore the perturbation of its outgoing zero-seeded tail is at most one.
// The three trajectories share all digit loads and can stop jointly at zero.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_segment_maps(__global const g3_digit* baseDigits,
                       __global const g3_i192* tails, __global const uchar* widths,
                       __global uint4* maps, __global uint* status) {
  const uint s = get_global_id(0), first = s * G3_SEGMENT;
  const g3_i192 tail = tails[(s + G3_SEGMENTS - 1) % G3_SEGMENTS];
  g3_i192 cm = g3_add(tail, g3_state(0)), cz = tail, cp = g3_add(tail, g3_state(2));
  for (uint i = 0; i < G3_SEGMENT; ++i) {
    const uint j = first+i;
    const g3_i192 d = g3_from_digit(baseDigits[j]);
    const uint width = widths[j];
    g3_digit ignored;
    cm = g3_split(g3_add(d, cm), width, &ignored);
    cz = g3_split(g3_add(d, cz), width, &ignored);
    cp = g3_split(g3_add(d, cp), width, &ignored);
    if (g3_equal(cm, g3_zero()) && g3_equal(cz, g3_zero()) && g3_equal(cp, g3_zero())) { break; }
  }
  maps[s] = (uint4)(g3_state_index(cm, status, s), g3_state_index(cz, status, s),
                    g3_state_index(cp, status, s), 0);
}

// Inclusive scan within groups of 256 segment maps; 32 group totals.
__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_scan_segments(__global const uint4* maps, __global uint4* prefixes,
                        __global uint4* totals) {
  __local uint4 scan[G3_SCAN];
  const uint s = get_global_id(0), lid = get_local_id(0);
  scan[lid] = maps[s];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = 1; offset < G3_SCAN; offset <<= 1) {
    const uint4 value = lid >= offset ? g3_compose(scan[lid-offset], scan[lid]) : scan[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  prefixes[s] = scan[lid];
  if (lid == G3_SCAN-1) { totals[get_group_id(0)] = scan[lid]; }
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_scan_groups(__global const uint4* totals, __global uint4* groupPrefixes,
                      __global uint* status) {
  __local uint4 scan[G3_SCAN];
  const uint lid = get_local_id(0);
  scan[lid] = lid < G3_GROUPS ? totals[lid] : g3_identity();
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint offset = 1; offset < G3_GROUPS; offset <<= 1) {
    const uint4 value = lid >= offset ? g3_compose(scan[lid-offset], scan[lid]) : scan[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    scan[lid] = value;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid < G3_GROUPS) { groupPrefixes[lid] = scan[lid]; }
  if (lid == 0) {
    const uint4 all = scan[G3_GROUPS-1];
    // Deterministic preference for zero when more than one cyclic solution exists.
    if (all.y == 1) { status[3] = 1; }
    else if (all.x == 0) { status[3] = 0; }
    else if (all.z == 2) { status[3] = 2; }
    else { status[3] = 1; g3_error(status, G3_NO_FIXED_POINT, 0); }
  }
}

__kernel __attribute__((reqd_work_group_size(256,1,1)))
void gold3_apply_carry(__global const g3_digit* baseDigits, __global const g3_i192* tails,
                      __global const uchar* widths, __global const uint4* prefixes,
                      __global const uint4* groupPrefixes, __global g3_digit* out,
                      __global uint* status) {
  const uint s = get_global_id(0), group = s / G3_SCAN, first = s * G3_SEGMENT;
  const uint groupState = group ? g3_eval(groupPrefixes[group-1], status[3]) : status[3];
  const uint state = s % G3_SCAN ? g3_eval(prefixes[s-1], groupState) : groupState;
  const uint expected = g3_eval(prefixes[s], groupState);
  g3_i192 carry = g3_add(tails[(s+G3_SEGMENTS-1) % G3_SEGMENTS], g3_state(state));
  for (uint i = 0; i < G3_SEGMENT; ++i) {
    if (g3_equal(carry, g3_zero())) {
      // Base digits are already balanced. Once the correction vanishes, no
      // width fetch or further 192-bit arithmetic can change the remainder.
      for (; i < G3_SEGMENT; ++i) { out[first+i] = baseDigits[first+i]; }
      break;
    }
    const uint j = first+i;
    g3_digit digit;
    carry = g3_split(g3_checked_add(g3_from_digit(baseDigits[j]), carry, status, s), widths[j], &digit);
    out[j] = digit;
  }
  if (!g3_equal(carry, g3_state(expected))) { g3_error(status, G3_EXIT_MISMATCH, s); }
}
