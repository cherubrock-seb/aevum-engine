# Aevum v0.3.46

Apple FFT3161 now executes the prepared GF61 `MUL_LOW` tail through the exact
staged height-transform and pair kernels already used by the generic GF61 path.
The prepared transform is read-only. Non-Apple dispatch remains the upstream
monolithic `tailMulGF61` specialization.
