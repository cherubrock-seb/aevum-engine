# Aevum v0.3.53 — Apple nonblocking queue pacing

On Apple OpenCL 1.2 the periodic queue threshold now calls `clFlush` and returns
without waiting. The old marker-enqueue and marker-poll path can serialize the
many global staging kernels used by the Apple FFT3161 compatibility path.

Correctness barriers are unchanged:

- blocking reads remain blocking;
- `Queue::finish()` still finishes the queue;
- explicit cross-queue marker synchronization remains available;
- non-Apple builds use the previous queue logic.

A/B diagnostic:

```bash
AEVUM_APPLE_QUEUE_MARKER_WAIT=1 <program>
```

Strict staged-kernel diagnostic, separate from queue pacing:

```bash
AEVUM_APPLE_STAGE_FINISH=1 <program>
```
