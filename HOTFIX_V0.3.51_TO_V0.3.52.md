# Hotfix Aevum v0.3.51 to v0.3.52

Apple `set_u32` constants now use the canonical compact-word upload path rather
than a raw device-buffer fill.  Linux, Windows and CUDA dispatch remain
unchanged.
