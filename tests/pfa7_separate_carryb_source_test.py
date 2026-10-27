#!/usr/bin/env python3
from pathlib import Path

src = Path("src/Gpu.cpp").read_text()

# The retained-width PFA7 path normally uses the canonical carryB + fftP
# sequence.  PR #159 also provides AEVUM_PFA7_LEAD_MODE=2 as an explicit
# experimental rollback to the fused bridge, so test the semantic guards
# independently instead of requiring one brittle multiline source literal.
assert "if (!haveCarryBridge) {" in src
assert "else if (fft.pfa_radix == 7 &&" in src
assert 'std::getenv("AEVUM_PFA7_LEAD_MODE")' in src
assert 'std::strcmp(std::getenv("AEVUM_PFA7_LEAD_MODE"), "2") == 0' in src
assert 'args.value("AEVUM_PFA7_SEPARATE_CARRYB", 1)' in src
assert """        carryB(out);
        fftP(buf1, out);
      } else {
        fftPCarryB(buf1, out);
      }""" in src
assert src.count("AEVUM_PFA7_SEPARATE_CARRYB") == 1

print("ok: PFA7 retained-width path keeps canonical carryB + fftP with explicit lead-mode rollback")
