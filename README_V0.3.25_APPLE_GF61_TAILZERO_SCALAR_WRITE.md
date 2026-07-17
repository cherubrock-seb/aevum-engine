# Aevum Engine v0.3.25 — Apple GF61 special-tail scalar final write

The M2 rejected the vector final writer after all preceding special-tail stages had loaded. The Apple-only final writer now emits one GF61 value per work-item using the exact stock `writeTailFusedLine` output address. No extra buffer is used and non-Apple execution is unchanged.
