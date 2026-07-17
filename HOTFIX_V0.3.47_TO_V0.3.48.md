# Hotfix Aevum v0.3.47 -> v0.3.48

This hotfix is an overlay for an Aevum Engine v0.3.47 source tree. It adds the
Apple square stage trace and the optional strict staged `clFinish` diagnostic.
It does not claim to correct the remaining square error.

Apply inside the directory that contains the Aevum v0.3.47 tree:

```bash
unzip -oq Hotfix-Aevum-v0.3.47-to-v0.3.48-Apple-Stage-Trace.zip \
  -d Aevum-Engine-0.3.47-macOS-Portable

mv Aevum-Engine-0.3.47-macOS-Portable \
   Aevum-Engine-0.3.48-macOS-Portable
```

Rebuild `build-engine/libaevum_engine.so` after applying the overlay.
