// Host test: GF61 csqq / csqaq must give the right square for every M61-count pair, including the pairs whose sum is >= 9.
//
// Those pairs used to reach a self-recursive call that reduced the input and retried (csqq(modM61(a), 2, 2)).  The CUDA compiler refuses to
// inline a self-recursive function, so every caller that reached it (tailSquare's onePairSq via csqa(ab, addin, 6)) spilled to local memory.
// The fallback now reassigns its own arguments.  The harness compiles the real math.cl on any OpenCL device and compares the results with a
// reference computed in 128-bit integers modulo 2^61 - 1.  A GF61 value with count n lies in [0, n * M61).
//
// usage: gf61_csq_counts_kernel_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const char* HARNESS = R"CL(
#include "base.cl"
#include "math.cl"
KERNEL(64) tcsqq(global ulong2* a, global ulong2* o, u32 xc, u32 yc) {
  u32 i = get_global_id(0);
  GF61 r = modM61(csqq(U2(a[i].x, a[i].y), xc, yc));
  o[i] = (ulong2)(r.x, r.y);
}
KERNEL(64) tcsqaq(global ulong2* a, global ulong2* c, global ulong2* o, u32 xc, u32 yc) {
  u32 i = get_global_id(0);
  GF61 r = modM61(csqaq(U2(a[i].x, a[i].y), U2(c[i].x, c[i].y), xc, yc));
  o[i] = (ulong2)(r.x, r.y);
}
)CL";

typedef unsigned long long u64_t;
typedef unsigned __int128 u128_t;
static const u64_t M61 = (1ull << 61) - 1;

static u64_t mulmod(u64_t a, u64_t b) {
  u128_t p = (u128_t) a * b;
  u64_t r = (u64_t) (p & M61) + (u64_t) (p >> 61);
  r = (r & M61) + (r >> 61);
  return r >= M61 ? r - M61 : r;
}
static u64_t addmod(u64_t a, u64_t b) { u64_t r = a % M61 + b % M61; return r >= M61 ? r - M61 : r; }
static u64_t submod(u64_t a, u64_t b) { a %= M61; b %= M61; return a >= b ? a - b : a + M61 - b; }

static u64_t rnd() { return ((u64_t) rand() << 40 ^ (u64_t) rand() << 20 ^ (u64_t) rand()); }
// A random value in [0, count * M61).
static u64_t rndBelow(u32 count) { return rnd() % ((u128_t) count * M61 > ~0ull ? ~0ull : (u64_t) count * M61); }

struct Harness {
  cl_device_id dev;
  cl_context ctx;
  cl_queue q;
  Program prog;
  cl_kernel kq = nullptr, kaq = nullptr;
};

static bool build(Harness& h, const std::string& dir) {
  h.prog = loadSource(h.ctx, HARNESS);
  std::string opts = "-cl-std=CL1.2 -I " + dir + " -DNO_ASM=1 -DAMDGPU=0 -DFFT_TYPE=3 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DWordSize=8u "
    "-DCARRY_LEN=8u -DPFA_RADIX=0u -DMAXBPW=3998u -DEXP=1257787u -DWIDTH=256u -DSMALL_HEIGHT=256u -DMIDDLE=2u -DNW=4u -DNH=4u -DFFT_VARIANT=101u "
    "-DFRAC_BPW_HI=1u -DFRAC_BPW_LO=4294967295u -DTAILTGF61=U2(1ul,1ul) -DDISTGF61=0ul -DDISTWTRIGGF61=0ul -DDISTMTRIGGF61=0ul -DDISTHTRIGGF61=0ul";
  if (clBuildProgram(h.prog.get(), 1, &h.dev, opts.c_str(), nullptr, nullptr) != 0) {
    std::string log = getBuildLog(h.prog.get(), h.dev);
    size_t e = log.find("error:");
    std::printf("FAIL: build: %s\n", (e == std::string::npos ? log.substr(0, 300) : log.substr(e, 300)).c_str());
    return false;
  }
  int err = 0;
  h.kq = clCreateKernel(h.prog.get(), "tcsqq", &err);
  CHECK1(err);
  h.kaq = clCreateKernel(h.prog.get(), "tcsqaq", &err);
  CHECK1(err);
  return true;
}

static cl_mem makeBuf(Harness& h, size_t bytes) {
  int err = 0;
  cl_mem b = clCreateBuffer(h.ctx, CL_MEM_READ_WRITE, bytes, nullptr, &err);
  CHECK1(err);
  return b;
}

// Runs csqq (withAdd == false) or csqaq (withAdd == true) for one (xc, yc) pair; returns the number of wrong results.
static size_t runCase(Harness& h, u32 xc, u32 yc, bool withAdd) {
  const size_t n = 2048;
  std::vector<u64_t> a(2 * n), c(2 * n), o(2 * n);
  for (size_t i = 0; i < n; ++i) {
    a[2 * i] = rndBelow(xc); a[2 * i + 1] = rndBelow(yc);
    c[2 * i] = rndBelow(2); c[2 * i + 1] = rndBelow(2);
  }
  a[0] = a[1] = 0;                                         // zero
  a[2] = xc * M61 - 1 > ~0ull ? ~0ull : xc * M61 - 1;     // top of the range (and not a canonical value)
  a[3] = yc * M61 - 1 > ~0ull ? ~0ull : yc * M61 - 1;
  cl_mem ba = makeBuf(h, n * 16), bc = makeBuf(h, n * 16), bo = makeBuf(h, n * 16);
  std::unique_ptr<cl_mem> baHold{ba}, bcHold{bc}, boHold{bo};
  CHECK1(clEnqueueWriteBuffer(h.q, ba, 1, 0, n * 16, a.data(), 0, nullptr, nullptr));
  CHECK1(clEnqueueWriteBuffer(h.q, bc, 1, 0, n * 16, c.data(), 0, nullptr, nullptr));
  cl_kernel k = withAdd ? h.kaq : h.kq;
  int arg = 0;
  CHECK1(clSetKernelArg(k, arg++, sizeof ba, &ba));
  if (withAdd) CHECK1(clSetKernelArg(k, arg++, sizeof bc, &bc));
  CHECK1(clSetKernelArg(k, arg++, sizeof bo, &bo));
  CHECK1(clSetKernelArg(k, arg++, sizeof xc, &xc));
  CHECK1(clSetKernelArg(k, arg++, sizeof yc, &yc));
  size_t g = n, l = 64;
  CHECK1(clEnqueueNDRangeKernel(h.q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
  CHECK1(clEnqueueReadBuffer(h.q, bo, 1, 0, n * 16, o.data(), 0, nullptr, nullptr));
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    u64_t x = a[2 * i], y = a[2 * i + 1];
    u64_t re = submod(mulmod(x % M61, x % M61), mulmod(y % M61, y % M61));
    u64_t im = mulmod(2, mulmod(x % M61, y % M61));
    if (withAdd) { re = addmod(re, c[2 * i]); im = addmod(im, c[2 * i + 1]); }
    if (o[2 * i] % M61 != re || o[2 * i + 1] % M61 != im) ++bad;
  }
  return bad;
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  Harness h;
  h.dev = devices[0];
  h.ctx = createContext(h.dev);
  h.q = makeQueue(h.dev, h.ctx, false);
  if (!build(h, dir)) return 1;
  srand(11);
  // Count pairs the kernels use (csqq 3,4 / csq 4,3 / csqa 6,6 in onePairSq), the neighbours of the sum-9 threshold, and larger sums that
  // take the reduce-first fallback (which continues with counts 2,2 on the reduced value).  Counts stay <= 7 so a value (< 7 * M61 < 2^64) fits a u64.
  const u32 pairs[][2] = {{1, 1}, {3, 4}, {4, 3}, {4, 4}, {3, 5}, {5, 3}, {4, 5}, {5, 4}, {6, 6}, {6, 3}, {3, 6}, {7, 7}, {2, 7}};
  bool ok = true;
  for (auto& p : pairs) {
    for (int withAdd = 0; withAdd < 2; ++withAdd) {
      size_t bad = runCase(h, p[0], p[1], withAdd);
      std::printf("%s(counts %u,%u): %zu of 2048 wrong  %s\n", withAdd ? "csqaq" : "csqq", p[0], p[1], bad, bad ? "FAIL" : "ok");
      ok &= bad == 0;
    }
  }
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
