# Aevum Engine v0.3.35 — Apple scalar pair multiply and middle-out

Apple OpenCL 1.2 only:

- scalar direct GF61 pair multiplication for special and normal lines;
- five-stage large-plan GF61 middle-out;
- no new transform-sized allocation;
- no changes to non-Apple arithmetic or queue behavior.

The Apple queue marker submission remains guarded by `#if defined(__APPLE__)`.
