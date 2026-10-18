// Host test: the GF61 onePairSq in tailsquare.cl must compute the exact pair squaring, mod M61, for both
// AEVUM_GF61_BETTER_ONEPAIRSQ=0 and =1.
//
// The =1 path saves modM61 reductions by tracking how far each unreduced intermediate can range (in multiples of
// M61).  A wrong range count wraps a u64 or a u128 product and gives a wrong value mod M61 only for some inputs,
// usually the extreme ones, so this test compiles the real tailsquare.cl and runs onePairSq on every combination of
// boundary values (0, 1, M61-1, M61, the largest modM61 output, ...) for the six input components, plus random
// inputs, for all four t_squared types, and compares each output mod M61 with an exact host model.
//
// usage: gf61_onepairsq_test [cl-source-dir]   (default src/cl)

#include "clwrap.h"
#include "common.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const char* HARNESS = R"CL(
#include "tailsquare.cl"

// in: 3 GF61 per case (a, b, t_squared); out: 2 GF61 per case (*pa, *pb after onePairSq).
kernel void pairsq(global const ulong2* in, global ulong2* out, uint type, uint n) {
  uint i = get_global_id(0);
  if (i >= n) return;
  GF61 a = in[3 * i], b = in[3 * i + 1], t = in[3 * i + 2];
  if (type == 0) onePairSq(&a, &b, t, 0);
  if (type == 1) onePairSq(&a, &b, t, 1);
  if (type == 2) onePairSq(&a, &b, t, 2);
  if (type == 3) onePairSq(&a, &b, t, 3);
  out[2 * i] = a;
  out[2 * i + 1] = b;
}
)CL";

using u64 = std::uint64_t;
using u128 = unsigned __int128;
static const u64 P = (u64(1) << 61) - 1;

struct G { u64 x, y; };
static u64 red(u64 a) { return a % P; }
static u64 add(u64 a, u64 b) { return (a + b) % P; }
static u64 sub(u64 a, u64 b) { return (a + P - b) % P; }
static u64 mul(u64 a, u64 b) { return u64((u128(a) * b) % P); }
static G gadd(G a, G b) { return {add(a.x, b.x), add(a.y, b.y)}; }
static G gsub(G a, G b) { return {sub(a.x, b.x), sub(a.y, b.y)}; }
static G gmul(G a, G b) { return {sub(mul(a.x, b.x), mul(a.y, b.y)), add(mul(a.x, b.y), mul(a.y, b.x))}; }
static G conj(G a) { return {a.x, sub(0, a.y)}; }
static G muli(G a) { return {sub(0, a.y), a.x}; }

// What onePairSq computes, from the !AEVUM_GF61_BETTER_ONEPAIRSQ code: X2(a, conjugate(b)), then
// c = a^2 - i^type * b^2 * t_squared, d = 2ab, X2(c, d), d = conjugate(d), and each result has its real and imaginary
// parts swapped.
static void model(G a, G b, G t, unsigned type, G* ra, G* rb) {
  a = {red(a.x), red(a.y)}; b = {red(b.x), red(b.y)}; t = {red(t.x), red(t.y)};
  G A = gadd(a, conj(b)), B = gsub(a, conj(b));
  G a2 = gmul(A, A), b2t2 = gmul(gmul(B, B), t), d = gmul(A, B);
  d = gadd(d, d);
  for (unsigned k = 0; k < type; ++k) b2t2 = muli(b2t2);
  G c = gsub(a2, b2t2);
  G c2 = gadd(c, d), d2 = conj(gsub(c, d));
  *ra = {c2.y, c2.x};
  *rb = {d2.y, d2.x};
}

int main(int argc, char** argv) {
  std::string dir = argc > 1 ? argv[1] : "src/cl";
  vector<cl_device_id> devices = getAllDeviceIDs();
  if (devices.empty()) { std::printf("SKIP: no OpenCL device\n"); return 0; }
  cl_device_id dev = devices[0];
  cl_context ctx = createContext(dev);
  cl_queue q = makeQueue(dev, ctx, false);

  // Values a reduced GF61 component can take: modM61 returns at most M61 + 7.
  const u64 edges[] = {0, 1, 2, P / 2, P - 2, P - 1, P, P + 7};
  const unsigned NE = sizeof(edges) / sizeof(edges[0]);
  std::vector<u64> in;
  for (unsigned i = 0; i < NE * NE * NE * NE * NE * NE; ++i) {
    unsigned k = i;
    for (int c = 0; c < 6; ++c) { in.push_back(edges[k % NE]); k /= NE; }
  }
  std::mt19937_64 rng(61);
  for (int i = 0; i < 65536; ++i) {
    for (int c = 0; c < 6; ++c) { u64 r = rng(); in.push_back((r & 15) == 0 ? P + (r >> 61) : r % P); }
  }
  // a and b equal (the pair of a line with itself), and b the conjugate of a.
  for (int i = 0; i < 4096; ++i) {
    u64 x = rng() % (P + 8), y = rng() % (P + 8), tx = rng() % P, ty = rng() % P;
    u64 v1[6] = {x, y, x, y, tx, ty}, v2[6] = {x, y, x, sub(0, red(y)), tx, ty};
    in.insert(in.end(), v1, v1 + 6);
    in.insert(in.end(), v2, v2 + 6);
  }
  const unsigned n = in.size() / 6;

  int err = 0;
  cl_mem bin = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in.size() * 8, in.data(), &err);
  CHECK1(err);
  cl_mem bout = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size_t(n) * 32, nullptr, &err);
  CHECK1(err);
  std::unique_ptr<cl_mem> binHold{bin}, boutHold{bout};

  bool ok = true;
  for (int better = 0; better <= 1; ++better) {
    Program prog = loadSource(ctx, HARNESS);
    std::string opts = "-I " + dir + " -DNO_ASM=1 -DFFT_FP64=0 -DFFT_FP32=0 -DNTT_GF31=0 -DNTT_GF61=1 -DFFT_TYPE=3 "
      "-DEXP=60000049u -DWIDTH=512u -DSMALL_HEIGHT=512u -DMIDDLE=8u -DNW=8u -DNH=8u -DWordSize=4u -DCARRY_LEN=8u "
      "-DPFA_RADIX=0u -DMAXBPW=3998u -DFFT_VARIANT=202u -DDISTGF61=0u -DDISTHTRIGGF61=0u -DAEVUM_GF61_BETTER_ONEPAIRSQ=" + std::to_string(better);
    if (clBuildProgram(prog.get(), 1, &dev, opts.c_str(), nullptr, nullptr) != 0) {
      std::printf("FAIL: build (AEVUM_GF61_BETTER_ONEPAIRSQ=%d)\n%s\n", better, getBuildLog(prog.get(), dev).c_str());
      return 1;
    }
    cl_kernel k = clCreateKernel(prog.get(), "pairsq", &err);
    CHECK1(err);
    KernelHolder kHold{k};
    std::vector<u64> out(size_t(n) * 4);
    for (unsigned type = 0; type < 4; ++type) {
      CHECK1(clSetKernelArg(k, 0, sizeof bin, &bin));
      CHECK1(clSetKernelArg(k, 1, sizeof bout, &bout));
      CHECK1(clSetKernelArg(k, 2, sizeof type, &type));
      CHECK1(clSetKernelArg(k, 3, sizeof n, &n));
      size_t global = (n + 63) / 64 * 64;
      CHECK1(clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
      CHECK1(clEnqueueReadBuffer(q, bout, 1, 0, out.size() * 8, out.data(), 0, nullptr, nullptr));
      unsigned bad = 0;
      for (unsigned i = 0; i < n; ++i) {
        const u64* v = &in[6 * size_t(i)];
        G ra, rb;
        model({v[0], v[1]}, {v[2], v[3]}, {v[4], v[5]}, type, &ra, &rb);
        const u64* o = &out[4 * size_t(i)];
        // A result must be a valid quick value for the next step (at most M61 + 7, as modM61 returns) and exact mod M61.
        bool good = o[0] <= P + 7 && o[1] <= P + 7 && o[2] <= P + 7 && o[3] <= P + 7 &&
                    red(o[0]) == ra.x && red(o[1]) == ra.y && red(o[2]) == rb.x && red(o[3]) == rb.y;
        if (!good && bad++ < 3) {
          std::printf("  mismatch BETTER=%d type %u: a=(%llx,%llx) b=(%llx,%llx) t=(%llx,%llx)\n", better, type,
                      (unsigned long long) v[0], (unsigned long long) v[1], (unsigned long long) v[2],
                      (unsigned long long) v[3], (unsigned long long) v[4], (unsigned long long) v[5]);
        }
      }
      std::printf("onePairSq GF61 AEVUM_GF61_BETTER_ONEPAIRSQ=%d type %u: %u of %u cases wrong  %s\n", better, type, bad, n,
                  bad ? "FAIL" : "ok");
      if (bad) ok = false;
    }
  }
  std::printf("%s\n", ok ? "ok" : "FAIL");
  return ok ? 0 : 1;
}
