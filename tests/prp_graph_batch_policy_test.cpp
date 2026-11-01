#include "../src/PrpGraphBatch.h"

#include <cassert>
#include <initializer_list>
#include <limits>

// Reference: visit the individual iterations and reject any that reaches an
// observable PRP boundary. This does not execute or emulate GPU arithmetic.
static bool scalarFits(std::uint64_t k, unsigned count, unsigned block,
                       std::uint64_t proof, std::uint64_t end,
                       std::uint64_t start, std::uint64_t requested) {
  if (count <= 1 || !block || k < start || k % block == 0 ||
      k > std::numeric_limits<std::uint64_t>::max() - count) return false;
  for (unsigned i = 1; i <= count; ++i) {
    const auto next = k + i;
    if (next % block == 0 || next >= proof || next >= end ||
        (requested && next - start >= requested)) return false;
  }
  return true;
}

int main() {
  for (unsigned n : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u})
    assert(parsePrpGraphBatch(std::to_string(n)) == n);
  for (const char* value : {"", "0", "3", "256", "-1", " 8", "08", "8x"}) {
    bool rejected = false;
    try { parsePrpGraphBatch(value); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
  }
  for (unsigned block : {1u, 17u, 200u, 1000u}) {
    for (unsigned count : {2u, 4u, 8u, 16u, 32u, 64u, 128u}) {
      for (std::uint64_t k = 1000; k < 3100; ++k) {
        for (std::uint64_t requested : {0u, 1777u})
          assert(prpGraphBatchFits(k, count, block, 2517, 3011, 1000, requested) ==
                 scalarFits(k, count, block, 2517, 3011, 1000, requested));
      }
    }
  }
  assert(!prpGraphBatchFits(100, 2, 0, 200, 200, 0, 0));
  assert(!prpGraphBatchFits(99, 2, 200, 200, 200, 100, 0));
  assert(prpGraphBatchFits(1001, 32, 1000, 1034, 2000, 1000, 0));
  assert(!prpGraphBatchFits(1001, 32, 1000, 1033, 2000, 1000, 0));
  // Resume, final-residue and requested-stop boundaries near counter overflow.
  const auto max = std::numeric_limits<std::uint64_t>::max();
  for (std::uint64_t k = max - 160; k < max; ++k)
    for (unsigned count : {2u, 32u, 128u})
      assert(prpGraphBatchFits(k, count, 1000, max, max, max - 200, 199) ==
             scalarFits(k, count, 1000, max, max, max - 200, 199));
}
