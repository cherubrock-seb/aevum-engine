// Host test: middleShuffle(lds, u, workgroupSize, blockSize) in fft-middle.cl must do the fftMiddleIn/Out partial transpose
// of the plain formula: thread t stores u[i] in slot (t % blockSize) * (workgroupSize / blockSize) + t / blockSize of row i,
// and thread me reads slot me back, so u[i] of thread me ends up holding what u[i] of thread
// (me % R) * blockSize + me / R held, R = workgroupSize / blockSize.  fftMiddleIn/Out then store u[i] at out[i * WG + me].
//
// The overloads pick LDS addresses that avoid bank conflicts for some workgroupSize / blockSize ratios and fall back
// to the plain transpose for the others, so a wrong address mapping corrupts only some shapes.  The test compiles the
// real fft-middle.cl for each type (FP64, FP32, GF31, GF61) at MIDDLE 8, 9 and 16, plus FP64 at 5 and FP32 at 7 (an odd
// MIDDLE other than FP64's is a PFA plan, compiled here with PFA_RADIX = MIDDLE; the GF31/GF61 radix-7 PFA needs more
// host constants than this harness supplies).  The FP64 and GF61 overloads move 32-bit pieces when MIDDLE > 8.  It runs
// middleShuffle on tagged values for every IN_WG / IN_SIZEX (OUT_WG / OUT_SIZEX) pair the tuner tries and checks every
// element.
//
// usage: middle_shuffle_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ELEM2 is the complex type middleShuffle permutes, ELEM its LDS element type, and BITS2 the same-sized integer vector
// the host uses to tag each value.
static const char* HARNESS = R"CL(
#include "base.cl"
#include "fft-middle.cl"

kernel void shuf(global const BITS2* in, global BITS2* out, uint wg, uint bs) {
  local ELEM lds[256 * MIDDLE * (sizeof(ELEM2) / sizeof(ELEM)) / 2];
  uint me = get_local_id(0);
  ELEM2 u[MIDDLE];
  for (uint i = 0; i < MIDDLE; ++i) u[i] = AS_ELEM2(in[i * wg + me]);
  middleShuffle(lds, u, wg, bs);
  for (uint i = 0; i < MIDDLE; ++i) out[i * wg + me] = AS_BITS2(u[i]);
}
)CL";

struct TypeCfg { const char* name; const char* defs; unsigned bytes; std::vector<unsigned> middles; };

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];

  char ext[8192] = {};
  clGetDeviceInfo(dev, 0x1030 /* CL_DEVICE_EXTENSIONS */, sizeof(ext) - 1, ext, nullptr);
  bool fp64 = std::string(ext).find("cl_khr_fp64") != std::string::npos;
  size_t maxWg = 0;
  clGetDeviceInfo(dev, 0x1004 /* CL_DEVICE_MAX_WORK_GROUP_SIZE */, sizeof maxWg, &maxWg, nullptr);

  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  const TypeCfg types[] = {
    {"FP64", "-DFFT_FP64=1 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=0 -DFFT_TYPE=0 -DELEM=T -DELEM2=T2 -DBITS2=ulong2 "
             "-DAS_ELEM2=as_double2 -DAS_BITS2=as_ulong2", 8, {5, 8, 9, 16}},
    {"FP32", "-DFFT_FP64=0 -DFFT_FP32=1 -DNTT_GF31=0 -DNTT_GF61=0 -DFFT_TYPE=53 -DELEM=F -DELEM2=F2 -DBITS2=uint2 "
             "-DAS_ELEM2=as_float2 -DAS_BITS2=as_uint2", 4, {7, 8, 9, 16}},
    {"GF31", "-DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=1 -DNTT_GF61=0 -DFFT_TYPE=52 -DELEM=Z31 -DELEM2=GF31 -DBITS2=uint2 "
             "-DAS_ELEM2=as_uint2 -DAS_BITS2=as_uint2", 4, {8, 9, 16}},
    {"GF61", "-DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DFFT_TYPE=3 -DELEM=Z61 -DELEM2=GF61 -DBITS2=ulong2 "
             "-DAS_ELEM2=as_ulong2 -DAS_BITS2=as_ulong2", 8, {8, 9, 16}},
  };
  // The IN_WG / IN_SIZEX and OUT_WG / OUT_SIZEX values tune.cpp tries: ratios 1:1, 2:1, 1:2 and the 4:1, 1:4, 8:1, 1:8
  // fallbacks.
  const unsigned wgs[] = {64, 128, 256}, sizexs[] = {8, 16, 32};

  bool ok = true;
  unsigned cases = 0, skipped = 0;
  for (const TypeCfg& t : types) {
    if (t.bytes == 8 && std::string(t.name) == "FP64" && !fp64) { std::printf("SKIP: %s, device has no double support\n", t.name); ++skipped; continue; }
    for (unsigned middle : t.middles) {
      Program prog = loadSource(ctx, HARNESS);
      std::string opts = "-I " + dir + " -DNO_ASM=1 " + t.defs + " -DEXP=17000023u -DWIDTH=512u -DSMALL_HEIGHT=512u -DMIDDLE=" +
        std::to_string(middle) + "u -DNW=8u -DNH=8u -DWordSize=4u -DCARRY_LEN=8u -DMAXBPW=3998u -DFFT_VARIANT=202u "
        // An odd MIDDLE other than FP64's 5 and 9 exists only as a PFA plan (pfa3/pfa7/pfa9), whose fftMiddleIn/Out use
        // this middleShuffle too.
        "-DPFA_RADIX=" + std::to_string(middle % 2 && std::string(t.name) != "FP64" ? middle : 0) + "u "
        "-DPFA_LOG2_ROOT_TWO31=1u -DPFA_LOG2_ROOT_TWO61=1u "     // required with PFA_RADIX, unused by middleShuffle
        // trig.cl needs the FP64/FP32 trig coefficients to compile; middleShuffle does not use them.
        "-DTRIG_SCALE=1 -DTRIG_SIN={0,0,0,0,0,0,0,0} -DTRIG_COS={0,0,0,0,0,0,0,0}";
      if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
        std::printf("FAIL: build %s MIDDLE=%u\n%s\n", t.name, middle, getBuildLog(prog.get(), dev).c_str());
        ok = false;
        continue;
      }
      int err = 0;
      cl_kernel k = clCreateKernel(prog.get(), "shuf", &err);
      CHECK1(err);
      KernelHolder kHold{k};
      size_t kMaxWg = 0;
      CHECK1(clGetKernelWorkGroupInfo(k, dev, 0x11B0 /* CL_KERNEL_WORK_GROUP_SIZE */, sizeof kMaxWg, &kMaxWg, nullptr));

      const size_t n = 256 * middle;            // complex values, at most
      const size_t words = n * 2;               // components
      std::vector<uint64_t> in64(words), out64(words);
      std::vector<uint32_t> in32(words), out32(words);
      cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, words * t.bytes, nullptr, &err);
      CHECK1(err);
      cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, words * t.bytes, nullptr, &err);
      CHECK1(err);
      std::unique_ptr<cl_mem> binHold{bin}, boutHold{bout};

      for (unsigned wg : wgs) {
        for (unsigned bs : sizexs) {
          if (wg > maxWg || (kMaxWg && wg > kMaxWg)) { ++skipped; continue; }
          // Tag value i * wg + me, component c.  For FP64/FP32 make the tag a small exact float so no bit pattern is a NaN.
          auto tag = [&](size_t idx, unsigned c) { return idx * 2 + c + 1; };
          for (size_t idx = 0; idx < size_t(wg) * middle; ++idx) {
            for (unsigned c = 0; c < 2; ++c) {
              uint64_t v = tag(idx, c);
              if (std::string(t.name) == "FP64") { double d = double(v); std::memcpy(&in64[idx * 2 + c], &d, 8); }
              else if (std::string(t.name) == "FP32") { float f = float(v); std::memcpy(&in32[idx * 2 + c], &f, 4); }
              else if (t.bytes == 8) in64[idx * 2 + c] = v | (v << 40);     // use the high half of each Z61 too
              else in32[idx * 2 + c] = uint32_t(v);
            }
          }
          const void* src = t.bytes == 8 ? (const void*) in64.data() : (const void*) in32.data();
          CHECK1(clEnqueueWriteBuffer(q, bin, 1, 0, size_t(wg) * middle * 2 * t.bytes, src, 0, nullptr, nullptr));
          CHECK1(clSetKernelArg(k, 0, sizeof bin, &bin));
          CHECK1(clSetKernelArg(k, 1, sizeof bout, &bout));
          CHECK1(clSetKernelArg(k, 2, sizeof wg, &wg));
          CHECK1(clSetKernelArg(k, 3, sizeof bs, &bs));
          size_t g = wg;
          CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &g, &g, 0, nullptr, nullptr));
          void* dst = t.bytes == 8 ? (void*) out64.data() : (void*) out32.data();
          CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, size_t(wg) * middle * 2 * t.bytes, dst, 0, nullptr, nullptr));

          unsigned bad = 0;
          for (unsigned me = 0; me < wg; ++me) {
            unsigned rows = wg / bs, from = (me % rows) * bs + me / rows;
            for (unsigned i = 0; i < middle; ++i) {
              size_t o = (size_t(i) * wg + me) * 2, e = (size_t(i) * wg + from) * 2;
              bool same = t.bytes == 8 ? (out64[o] == in64[e] && out64[o + 1] == in64[e + 1])
                                       : (out32[o] == in32[e] && out32[o + 1] == in32[e + 1]);
              if (!same) ++bad;
            }
          }
          ++cases;
          if (bad) {
            ok = false;
            std::printf("%s MIDDLE=%2u WG=%3u SIZEX=%2u: %u of %u values misplaced  FAIL\n", t.name, middle, wg, bs, bad, wg * middle);
          }
        }
      }
      std::printf("%s MIDDLE=%2u: shapes checked\n", t.name, middle);
    }
  }
  std::printf("middleShuffle: %u cases run, %u skipped\n", cases, skipped);
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
