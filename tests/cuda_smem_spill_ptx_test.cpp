#include "cuda/PtxRegisterCache.h"
#include "cuda/SmemSpillPtx.h"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  const std::string neutral = R"(.version 9.0
.target sm_86
.address_size 64
.visible .entry tailSquareGF61(
 .param .u64 p0
)
.maxntid 64,1,1
{
 .reg .b32 r;
 ret;
}
.visible .entry carryFused()
.maxntid 256,1,1
{
 ret;
}
)";

  const auto reg = aevumTakePtxRegisterLimit(
      "--aevum-reg-ptx=1 --aevum-smemspill=1 --maxrregcount=80 -DTEST=1 ");
  assert(reg.enabled && reg.maxRegs == 80 && reg.minBlocks == 0);
  const auto spill = aevumTakeSmemSpillMarker(reg.compileArgs);
  assert(spill.enabled);
  assert(spill.compileArgs.find("aevum-smemspill") == std::string::npos);
  assert(spill.compileArgs.find("aevum-reg-ptx") == std::string::npos);
  assert(spill.compileArgs.find("--maxrregcount") == std::string::npos);
  assert(spill.compileArgs.find("-DTEST=1") != std::string::npos);
  assert(spill.compileArgs.find("-DAEVUM_CUDA_REG_PTX_BASE=1") != std::string::npos);

  const std::string limited =
      aevumPtxWithRegisterLimit(neutral, "tailSquareGF61", 80, 0);
  const std::string combined =
      aevumPtxWithSmemSpill(limited, "tailSquareGF61");

  assert(combined.find(".maxnreg 80") != std::string::npos);
  assert(combined.find(".pragma \"enable_smem_spilling\";") != std::string::npos);
  const auto carry = combined.find(".entry carryFused");
  assert(carry != std::string::npos);
  assert(combined.find("enable_smem_spilling", carry) == std::string::npos);

  const std::string again =
      aevumPtxWithSmemSpill(combined, "tailSquareGF61");
  assert(again == combined);

  const auto direct = aevumTakeSmemSpillMarker(
      "--aevum-smemspill=1 --maxrregcount=96 -DOTHER=1 ");
  assert(direct.enabled);
  assert(direct.compileArgs.find("--maxrregcount=96") != std::string::npos);

  bool rejected = false;
  try {
    (void)aevumTakeSmemSpillMarker(
        "--aevum-smemspill=1 --aevum-smemspill=1 ");
  } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);

  std::string old = neutral;
  old.replace(old.find("9.0"), 3, "8.8");
  rejected = false;
  try {
    (void)aevumPtxWithSmemSpill(old, "tailSquareGF61");
  } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);

  rejected = false;
  try {
    (void)aevumPtxWithSmemSpill(neutral, "missing");
  } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);

  std::cout
      << "shared-spill marker, PTX9 named-entry rewrite and register-limit composition PASS\n";
}
