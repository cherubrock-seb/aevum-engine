# Aevum Engine v0.3.24 — Apple GF61 special-tail PairApple argument fix

The v0.3.23 host code fixed `bufTrigH` at kernel argument 1 and later invoked the kernel with `(scratch, bank)`. `Kernel::operator()` binds dynamic arguments from position zero, so the 4-byte bank value overwrote the `cl_mem` expected at argument 1.

v0.3.24 removes the fixed middle argument and invokes `tailSquareZeroGF61PairApple` with all three arguments explicitly: `(scratch, bufTrigH, bank)`. No OpenCL source or non-Apple path changes.
