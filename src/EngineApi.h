/*
Aevum register API adaptation, Copyright 2026 cherubrock-seb.
Derived from GPUOwl/PRPLL by Mihai Preda and George Woltman.
Licensed under GNU GPL version 3. See LICENSE and UPSTREAM.md.
*/

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(AEVUM_ENGINE_LIBRARY)
    #define AEVUM_ENGINE_API __declspec(dllexport)
  #else
    #define AEVUM_ENGINE_API __declspec(dllimport)
  #endif
#else
  #define AEVUM_ENGINE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* aevum_engine_handle;

// Keep numeric values aligned with PrMers engine::gpu_workload.  The old
// create symbol remains ABI-compatible; create_ex is optional for older hosts.
enum aevum_engine_workload {
  AEVUM_WORKLOAD_GENERIC = 0,
  AEVUM_WORKLOAD_PRP = 1,
  AEVUM_WORKLOAD_LL = 2,
  AEVUM_WORKLOAD_PM1 = 3,
  AEVUM_WORKLOAD_PM1_LOWMEM = 4,
  AEVUM_WORKLOAD_PM1_ULTRALOWMEM = 5,
  AEVUM_WORKLOAD_ECM = 6,
  // Explicit candidate arithmetic only. Never use for a production PRP/LL run.
  // Requires fft_spec="gold64-validation:512:8:512:202" and OpenCL.
  AEVUM_WORKLOAD_GOLD64_VALIDATION = 0x1000
};

AEVUM_ENGINE_API const char* aevum_engine_version(void);
AEVUM_ENGINE_API const char* aevum_engine_last_error(void);
AEVUM_ENGINE_API int aevum_engine_resolve_fft(
    uint32_t exponent,
    const char* fft_spec,
    char* output,
    size_t output_size);

AEVUM_ENGINE_API aevum_engine_handle aevum_engine_create(
    uint32_t exponent,
    size_t register_count,
    uint32_t device,
    int verbose,
    const char* fft_spec,
    const char* tune_dir);

AEVUM_ENGINE_API aevum_engine_handle aevum_engine_create_ex(
    uint32_t exponent,
    size_t register_count,
    uint32_t device,
    int verbose,
    const char* fft_spec,
    const char* tune_dir,
    uint32_t workload);

AEVUM_ENGINE_API void aevum_engine_destroy(aevum_engine_handle handle);
AEVUM_ENGINE_API size_t aevum_engine_transform_size(aevum_engine_handle handle);
AEVUM_ENGINE_API size_t aevum_engine_word_count(aevum_engine_handle handle);
// Copies the FFT plan the engine was actually created with (e.g. "1:1K:16:256:101") into output.
// This can differ from what aevum_engine_resolve_fft() predicts: autotune, tune.txt replay and device
// specific profiles are applied at creation.  Returns 1 on success, 0 if the buffer is too small.
AEVUM_ENGINE_API int aevum_engine_plan_spec(aevum_engine_handle handle, char* output, size_t output_size);
AEVUM_ENGINE_API int aevum_engine_sync(aevum_engine_handle handle);

// Opt-in production timing decomposition. Counters are inactive unless
// AEVUM_TIMING_DECOMP=1, so normal production runs pay no clock-sampling cost.
typedef struct aevum_engine_timing_stats {
  uint64_t square_calls;
  uint64_t pending_flush_calls;
  uint64_t pending_flush_ns;
  uint64_t queue_sync_calls;
  uint64_t queue_sync_ns;
  uint64_t copy_calls;
  uint64_t copy_ns;
  uint64_t equal_calls;
  uint64_t equal_ns;
  uint64_t readback_calls;
  uint64_t readback_ns;
} aevum_engine_timing_stats;
AEVUM_ENGINE_API int aevum_engine_timing_reset(aevum_engine_handle handle);
AEVUM_ENGINE_API int aevum_engine_timing_get(aevum_engine_handle handle, aevum_engine_timing_stats* stats);

// Diagnostic: synchronize, optionally emit kernel event counters, then reset them.
AEVUM_ENGINE_API int aevum_engine_profile_report(aevum_engine_handle handle, int emit);

AEVUM_ENGINE_API int aevum_engine_set_u32(aevum_engine_handle handle, size_t dst, uint32_t value);
AEVUM_ENGINE_API int aevum_engine_set_words(aevum_engine_handle handle, size_t dst, const uint32_t* words, size_t count);
AEVUM_ENGINE_API int aevum_engine_get_words(aevum_engine_handle handle, size_t src, uint32_t* words, size_t count);
AEVUM_ENGINE_API int aevum_engine_copy(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_prepare(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_square_mul(aevum_engine_handle handle, size_t reg, uint32_t factor);
AEVUM_ENGINE_API int aevum_engine_mul(aevum_engine_handle handle, size_t dst, size_t src, uint32_t factor);
AEVUM_ENGINE_API int aevum_engine_add(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_sub_reg(aevum_engine_handle handle, size_t dst, size_t src);
AEVUM_ENGINE_API int aevum_engine_sub_u32(aevum_engine_handle handle, size_t dst, uint32_t value);
AEVUM_ENGINE_API int aevum_engine_equal(aevum_engine_handle handle, size_t lhs, size_t rhs, int* equal_out);

// Diagnostic only: hashes GF31/GF61 after fftP, middle-in, tail-square,
// middle-out and fftW, followed by the final word-buffer hash and low word.
// trace_count must be at least 12. The source register is preserved.
AEVUM_ENGINE_API int aevum_engine_debug_square_trace(
    aevum_engine_handle handle, size_t src, uint64_t* trace, size_t trace_count);

// Diagnostic only: for an explicit Type1 FFT3161 handle, execute the square
// transform through fftW but before carry and return N signed exact CRT
// coefficients as N pairs of two's-complement uint64 limbs (lo64, hi64).
// The source register is preserved; limb_count must equal 2*transform_size.
AEVUM_ENGINE_API int aevum_engine_debug_type1_precarry(
    aevum_engine_handle handle, size_t src, uint64_t* limbs, size_t limb_count);

// Available only on the explicit Gold64 validation handle. Success of a
// register operation means a candidate was produced, NOT mathematical proof.
// Every report retains UNPROVEN_PRECISION. first_fatal_block is UINT32_MAX
// when no fatal block exists; blocks contain 256 natural-order digits.
enum aevum_gold64_status {
  AEVUM_GOLD64_UNPROVEN_PRECISION = 1, AEVUM_GOLD64_NONFINITE = 2,
  AEVUM_GOLD64_QUOTIENT_RANGE = 4, AEVUM_GOLD64_FRACTIONAL_HINT = 8,
  AEVUM_GOLD64_INVALID_RESIDUE = 16, AEVUM_GOLD64_UNSUPPORTED_OPERATION = 32,
  AEVUM_GOLD64_CARRY_OVERFLOW = 64, AEVUM_GOLD64_CARRY_NONCONVERGENCE = 128
};
typedef struct aevum_gold64_diagnostics {
  uint32_t flags;
  uint32_t first_fatal_block;
  uint32_t fatal_blocks;
  uint32_t unproven_blocks;
} aevum_gold64_diagnostics;
AEVUM_ENGINE_API int aevum_engine_gold64_diagnostics(
    aevum_engine_handle handle, aevum_gold64_diagnostics* report);

// Diagnostic only. Recomputes quotient selection in FP64 from the SAME FP32
// estimate and Gold64 residue. It never changes reconstruction, acceptance,
// register contents, or UNPROVEN_PRECISION. UINT32_MAX means no such index.
typedef struct aevum_gold64_quotient_diagnostics {
  uint32_t coefficients;
  uint32_t fp32_fractional;
  uint32_t fp64_fractional;
  uint32_t quotient_disagreements;
  uint32_t fp64_range;
  uint32_t first_disagreement;
  uint32_t first_fp32_fractional;
  uint32_t first_fp64_fractional;
} aevum_gold64_quotient_diagnostics;
AEVUM_ENGINE_API int aevum_engine_gold64_quotient_diagnostics(
    aevum_engine_handle handle, aevum_gold64_quotient_diagnostics* report);

// Diagnostic only: read the retained FP32 uncarried coefficient estimates from
// the most recent Gold64 square/multiply attempt. Count must be 4194304.
AEVUM_ENGINE_API int aevum_engine_gold64_get_estimates(
    aevum_engine_handle handle, float* estimates, size_t count);

// Diagnostic only: read the retained natural-order Gold64 modular residues from
// the most recent square/multiply attempt, before FP32 quotient reconstruction.
AEVUM_ENGINE_API int aevum_engine_gold64_get_residues(
    aevum_engine_handle handle, uint64_t* residues, size_t count);

// Diagnostic only: normalize a scratch candidate using FP64 quotient arithmetic
// over the same retained FP32 estimate/residue from the most recent square or
// multiply. The live register and published fail-stop diagnostics are unchanged.
AEVUM_ENGINE_API int aevum_engine_gold64_fp64_candidate_words(
    aevum_engine_handle handle, uint32_t factor, uint32_t* words, size_t count);

AEVUM_ENGINE_API int aevum_engine_gold64_get_digits(
    aevum_engine_handle handle, size_t src, int64_t* digits, size_t count);

// Diagnostic only: set exactly 4194304 natural-order signed digits without
// compact-word normalization. Available only on the Gold64 validation handle.
AEVUM_ENGINE_API int aevum_engine_gold64_set_raw_digits(
    aevum_engine_handle handle, size_t dst, const int64_t* digits, size_t count);

AEVUM_ENGINE_API int aevum_engine_gold64_square_minus_two(
    aevum_engine_handle handle, size_t reg);

#ifdef __cplusplus
}
#endif

