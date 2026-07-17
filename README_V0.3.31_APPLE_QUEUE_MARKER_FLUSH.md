# Aevum Engine v0.3.31 — Apple queue-marker flush

`Queue::queueMarkerEvent()` now calls `clFlush` after enqueuing the marker and before polling its event. This prevents Apple OpenCL from retaining the marker and all preceding staged kernels indefinitely in an unsubmitted command queue. Arithmetic code is unchanged.
