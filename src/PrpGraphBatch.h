#pragma once

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

// Host-only option: one preserves the existing per-square submission path.
inline unsigned parsePrpGraphBatch(std::string_view value) {
  for (unsigned n : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
    if (value == std::to_string(n)) return n;
  }
  throw std::runtime_error("AEVUM_CUDA_GRAPH_BATCH must be 1, 2, 4, 8, 16, 32, 64 or 128");
}

// k is the last completed iteration. Leave each boundary iteration to the
// ordinary PRP loop, including its check update, canonical output and stop logic.
// Subtraction avoids overflow near the end of the iteration counter.
inline bool prpGraphBatchFits(std::uint64_t k, unsigned count, unsigned blockSize,
                             std::uint64_t persistK, std::uint64_t kEnd,
                             std::uint64_t startK, std::uint64_t requestedIterations) {
  if (count <= 1 || !blockSize || k < startK || k % blockSize == 0) return false;
  if (count >= blockSize - k % blockSize) return false;
  if (k >= persistK || count >= persistK - k || k >= kEnd || count >= kEnd - k) return false;
  if (requestedIterations &&
      (k - startK >= requestedIterations || count >= requestedIterations - (k - startK))) return false;
  return true;
}
