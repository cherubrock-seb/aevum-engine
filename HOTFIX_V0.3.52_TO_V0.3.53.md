# Aevum hotfix v0.3.52 -> v0.3.53

Apple-only performance change: periodic queue pacing uses nonblocking `clFlush`
instead of marker polling. Set `AEVUM_APPLE_QUEUE_MARKER_WAIT=1` to restore the
old policy for diagnosis. Arithmetic kernels and all non-Apple runtime paths are unchanged.
