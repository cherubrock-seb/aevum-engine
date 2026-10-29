// The CUDA shim releases each object in the context that owns it, not in whichever context it made current last.
//
// With several contexts alive (several devices or workers, autotune contexts, engines created and destroyed through
// the engine API), clReleaseMemObject used to make the newest context current and cuMemFree there, ignoring the
// result.  This test builds the real src/cuda/clwrap_cuda.cpp against a recording stand-in for the CUDA driver
// (tests/cuda_stub/cuda.h and the cu* functions below): no GPU and no CUDA install needed.  The stand-in keeps a
// per-thread context stack like the driver, and fails a free, a stream or an event destroyed in the wrong context the
// way the driver can (CUDA_ERROR_INVALID_VALUE / CUDA_ERROR_INVALID_CONTEXT).
#include "cuda/tinycuda.h"
#include "cuda/cudawrap.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

bool prpll_verbose = false;
std::string NvrtcProgram::preprocessOpenCL(const std::string& source) { return source; }
std::string NvrtcProgram::compile(const std::string&, const std::string&, const std::vector<std::string>&,
                                  const std::vector<std::pair<std::string, std::string>>&) { return {}; }
const char* aevumCudaLastError();

// ---- The stand-in driver ----

struct CUctx_st { int id; bool alive; };
struct CUstream_st { CUcontext owner; };
struct CUevent_st { CUcontext owner; };

namespace {
std::vector<CUcontext> stack;                   // this thread's context stack; back() is current
std::map<CUdeviceptr, CUcontext> allocations;   // live device memory and the context it was allocated in
CUdeviceptr nextPtr = 0x10000;

struct Call { std::string what; CUcontext current; CUcontext owner; };
std::vector<Call> calls;                        // every free / destroy, with the context current at the time

CUresult failNextFree = CUDA_SUCCESS;           // make the next cuMemFree fail with this

CUcontext current() { return stack.empty() ? nullptr : stack.back(); }
bool alive(CUcontext c) { return c && c->alive; }

// Like the driver: an object can only be freed or destroyed with its own (live) context current.
CUresult destroyIn(const char* what, CUcontext owner) {
  calls.push_back({what, current(), owner});
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  return current() == owner ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
}  // namespace

CUresult cuInit(unsigned) { return CUDA_SUCCESS; }
CUresult cuGetErrorName(CUresult r, const char** s) { *s = r == CUDA_SUCCESS ? "CUDA_SUCCESS" : "CUDA_ERROR_STUB"; return CUDA_SUCCESS; }
CUresult cuDriverGetVersion(int* v) { *v = 12000; return CUDA_SUCCESS; }
CUresult cuDeviceGetCount(int* n) { *n = 1; return CUDA_SUCCESS; }
CUresult cuDeviceGet(CUdevice* d, int i) { *d = i; return CUDA_SUCCESS; }
CUresult cuDeviceGetName(char* name, int len, CUdevice) { std::snprintf(name, len, "stub"); return CUDA_SUCCESS; }
CUresult cuDeviceGetAttribute(int* v, CUdevice_attribute, CUdevice) { *v = 0; return CUDA_SUCCESS; }
CUresult cuDeviceTotalMem(size_t* b, CUdevice) { *b = 0; return CUDA_SUCCESS; }

CUresult cuCtxCreate(CUcontext* out, unsigned, CUdevice) {
  static int ids = 0;
  *out = new CUctx_st{++ids, true};
  stack.push_back(*out);   // cuCtxCreate pushes the new context
  return CUDA_SUCCESS;
}
CUresult cuCtxDestroy(CUcontext c) {
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  c->alive = false;        // kept allocated so a stale handle stays distinguishable
  stack.erase(std::remove(stack.begin(), stack.end(), c), stack.end());
  for (auto it = allocations.begin(); it != allocations.end();) {
    it = it->second == c ? allocations.erase(it) : std::next(it);
  }
  return CUDA_SUCCESS;
}
CUresult cuCtxSetCurrent(CUcontext c) {
  if (!c) { if (!stack.empty()) stack.pop_back(); return CUDA_SUCCESS; }
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  if (stack.empty()) stack.push_back(c); else stack.back() = c;
  return CUDA_SUCCESS;
}
CUresult cuCtxGetCurrent(CUcontext* c) { *c = current(); return CUDA_SUCCESS; }
CUresult cuCtxPushCurrent(CUcontext c) {
  if (!alive(c)) return CUDA_ERROR_INVALID_CONTEXT;
  stack.push_back(c);
  return CUDA_SUCCESS;
}
CUresult cuCtxPopCurrent(CUcontext* c) {
  if (stack.empty()) return CUDA_ERROR_INVALID_CONTEXT;
  if (c) *c = stack.back();
  stack.pop_back();
  return CUDA_SUCCESS;
}
CUresult cuCtxSetLimit(CUlimit, size_t) { return CUDA_SUCCESS; }

CUresult cuMemAlloc(CUdeviceptr* p, size_t) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  *p = nextPtr;
  nextPtr += 0x1000;
  allocations[*p] = current();
  return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr p) {
  auto it = allocations.find(p);
  CUresult r = destroyIn("cuMemFree", it == allocations.end() ? nullptr : it->second);
  if (r == CUDA_SUCCESS && failNextFree != CUDA_SUCCESS) r = failNextFree;
  failNextFree = CUDA_SUCCESS;
  if (r == CUDA_SUCCESS) allocations.erase(it);
  return r;
}
CUresult cuMemGetInfo(size_t* f, size_t* t) { *f = *t = 0; return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD(CUdeviceptr, const void*, size_t) { return CUDA_SUCCESS; }
CUresult cuMemcpyDtoH(void*, CUdeviceptr, size_t) { return CUDA_SUCCESS; }
CUresult cuMemcpyHtoDAsync(CUdeviceptr, const void*, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemcpyDtoHAsync(void*, CUdeviceptr, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemcpyDtoDAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemsetD8Async(CUdeviceptr, unsigned char, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemsetD16Async(CUdeviceptr, unsigned short, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemsetD32Async(CUdeviceptr, unsigned, size_t, CUstream) { return CUDA_SUCCESS; }
CUresult cuMemsetD2D32Async(CUdeviceptr, size_t, unsigned, size_t, size_t, CUstream) { return CUDA_SUCCESS; }

CUresult cuModuleLoadData(CUmodule*, const void*) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuModuleLoadDataEx(CUmodule*, const void*, unsigned, CUjit_option*, void**) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuModuleUnload(CUmodule) { return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction*, CUmodule, const char*) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuFuncGetAttribute(int* v, CUfunction_attribute, CUfunction) { *v = 0; return CUDA_SUCCESS; }
CUresult cuLaunchKernel(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream,
                        void**, void**) { return CUDA_SUCCESS; }

CUresult cuStreamCreate(CUstream* s, unsigned) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  *s = new CUstream_st{current()};
  return CUDA_SUCCESS;
}
CUresult cuStreamDestroy(CUstream s) {
  CUresult r = destroyIn("cuStreamDestroy", s->owner);
  if (r == CUDA_SUCCESS) delete s;
  return r;
}
CUresult cuStreamSynchronize(CUstream) { return CUDA_SUCCESS; }
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned) { return CUDA_SUCCESS; }
CUresult cuStreamSetAttribute(CUstream, CUstreamAttrID, const CUstreamAttrValue*) { return CUDA_SUCCESS; }

CUresult cuEventCreate(CUevent* e, unsigned) {
  if (!alive(current())) return CUDA_ERROR_INVALID_CONTEXT;
  *e = new CUevent_st{current()};
  return CUDA_SUCCESS;
}
CUresult cuEventDestroy(CUevent e) {
  CUresult r = destroyIn("cuEventDestroy", e->owner);
  if (r == CUDA_SUCCESS) delete e;
  return r;
}
CUresult cuEventRecord(CUevent, CUstream) { return CUDA_SUCCESS; }
CUresult cuEventQuery(CUevent) { return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent) { return CUDA_SUCCESS; }
CUresult cuEventElapsedTime(float* ms, CUevent, CUevent) { *ms = 0; return CUDA_SUCCESS; }

CUresult cuGraphCreate(CUgraph*, unsigned) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuGraphDestroy(CUgraph) { return CUDA_SUCCESS; }
CUresult cuGraphAddKernelNode(CUgraphNode*, CUgraph, const CUgraphNode*, size_t, const CUDA_KERNEL_NODE_PARAMS*) {
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuGraphInstantiateWithFlags(CUgraphExec*, CUgraph, unsigned long long) { return CUDA_ERROR_INVALID_VALUE; }
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS*) {
  return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuGraphExecDestroy(CUgraphExec) { return CUDA_SUCCESS; }
CUresult cuGraphLaunch(CUgraphExec, CUstream) { return CUDA_ERROR_INVALID_VALUE; }

nvrtcResult nvrtcVersion(int* major, int* minor) { *major = 12; *minor = 0; return NVRTC_SUCCESS; }
nvrtcResult nvrtcGetNumSupportedArchs(int* n) { *n = 0; return NVRTC_SUCCESS; }
nvrtcResult nvrtcGetSupportedArchs(int*) { return NVRTC_SUCCESS; }

// ---- The test ----

static int failures = 0;
static void require(bool ok, const char* what) {
  if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

// The single free/destroy recorded since the last clear, if it is `what`.
static const Call* only(const char* what) {
  return calls.size() == 1 && calls[0].what == what ? &calls[0] : nullptr;
}

static cl_context newContext(cl_device_id dev) {
  int err = -1;
  cl_context c = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  require(c && err == CL_SUCCESS, "clCreateContext");
  return c;
}

int main() {
  cl_device_id dev{};
  unsigned n = 0;
  require(clGetDeviceIDs(nullptr, 0, 1, &dev, &n) == CL_SUCCESS && n == 1, "clGetDeviceIDs");

  // An engine's context, then a second one (another engine, worker or an autotune context) made after it: the shim's
  // newest, and current.
  cl_context c1 = newContext(dev);
  int err = -1;
  cl_mem buf = clCreateBuffer(c1, CL_MEM_READ_WRITE, 4096, nullptr, &err);
  require(buf && err == CL_SUCCESS, "clCreateBuffer in the first context");
  cl_context c2 = newContext(dev);
  require(current() == c2->ctx, "the newest context is current");

  // A buffer of the first context, allocated while the second is the shim's newest, lives in the first.
  cl_mem buf2 = clCreateBuffer(c1, CL_MEM_READ_WRITE, 4096, nullptr, &err);
  require(buf2 && err == CL_SUCCESS, "clCreateBuffer in the first context while the second is newest");
  require(buf2 && allocations.count(buf2->ptr) && allocations[buf2->ptr] == c1->ctx,
          "a buffer is allocated in the context it is created for, not the shim's newest");

  // Releasing the first context's buffer frees it with the first context current, and leaves the caller's current.
  calls.clear();
  require(clReleaseMemObject(buf) == CL_SUCCESS, "clReleaseMemObject of the first context's buffer succeeds");
  const Call* free1 = only("cuMemFree");
  require(free1 && free1->current == c1->ctx, "the buffer is freed with its own context current, not the newest");
  require(current() == c2->ctx, "releasing a buffer restores the caller's current context");

  // A failing cuMemFree is reported, not swallowed; the caller's context is still restored.
  calls.clear();
  failNextFree = CUDA_ERROR_UNKNOWN;
  require(clReleaseMemObject(buf2) != CL_SUCCESS, "a cuMemFree error is returned by clReleaseMemObject");
  require(aevumCudaLastError() && std::strstr(aevumCudaLastError(), "(999)"), "the CUDA error is kept for the message");
  require(only("cuMemFree") != nullptr, "the failing release tried exactly one free");
  require(current() == c2->ctx, "a failed release restores the caller's current context");

  require(clReleaseMemObject(nullptr) == CL_SUCCESS, "releasing a null buffer is a no-op");

  // A command queue's stream is destroyed in the queue's context.
  cl_command_queue q = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);
  require(q && err == CL_SUCCESS, "clCreateCommandQueueWithProperties on the first context");
  cuCtxSetCurrent(c2->ctx);   // as ensureContextCurrent() leaves it: the newest
  calls.clear();
  require(clReleaseCommandQueue(q) == CL_SUCCESS, "clReleaseCommandQueue of the first context's queue succeeds");
  const Call* stream = only("cuStreamDestroy");
  require(stream && stream->current == c1->ctx, "the stream is destroyed with its own context current");
  require(current() == c2->ctx, "releasing a queue restores the caller's current context");

  // An event is destroyed in the context it was made in.
  cl_command_queue q1 = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);   // leaves c1 current
  cl_event ev = nullptr;
  require(clEnqueueMarkerWithWaitList(q1, 0, nullptr, &ev) == CL_SUCCESS && ev, "a marker event in the first context");
  cuCtxSetCurrent(c2->ctx);
  calls.clear();
  require(clReleaseEvent(ev) == CL_SUCCESS, "clReleaseEvent");
  const Call* event = only("cuEventDestroy");
  require(event && event->current == c1->ctx, "the event is destroyed with its own context current");
  require(current() == c2->ctx, "releasing an event restores the caller's current context");

  // SVM memory, given its context, is allocated and freed there.
  void* svm = clSVMAlloc(c1, 0, 256, 0);
  require(svm && allocations.count((CUdeviceptr)(uintptr_t)svm) && allocations[(CUdeviceptr)(uintptr_t)svm] == c1->ctx,
          "clSVMAlloc allocates in the given context");
  calls.clear();
  clSVMFree(c1, svm);
  const Call* svmFree = only("cuMemFree");
  require(svmFree && svmFree->current == c1->ctx, "clSVMFree frees in the given context");
  require(current() == c2->ctx, "clSVMFree restores the caller's current context");

  // A buffer released after its context was destroyed: cuCtxDestroy freed the memory, so nothing is freed now (above
  // all not in whichever context is current), and the release reports the dead context.
  cl_mem orphan = clCreateBuffer(c1, CL_MEM_READ_WRITE, 64, nullptr, &err);
  cl_command_queue orphanQueue = clCreateCommandQueueWithProperties(c1, dev, nullptr, &err);
  cuCtxSetCurrent(c2->ctx);
  require(orphan && orphanQueue, "a buffer and a queue of the first context");
  require(clReleaseCommandQueue(q1) == CL_SUCCESS, "clReleaseCommandQueue");
  clReleaseContext(c1);
  require(current() == c2->ctx, "the surviving context is current after the first is released");
  calls.clear();
  require(clReleaseMemObject(orphan) != CL_SUCCESS, "releasing a buffer of a destroyed context reports it");
  require(clReleaseCommandQueue(orphanQueue) != CL_SUCCESS, "releasing a queue of a destroyed context reports it");
  require(calls.empty(), "nothing of a destroyed context is freed or destroyed again, in any context");
  require(current() == c2->ctx, "releases after a context is destroyed leave the caller's context current");

  // Creating a buffer for a destroyed context fails instead of allocating in whichever context is current.
  cl_mem buf3 = clCreateBuffer(c2, CL_MEM_READ_WRITE, 64, nullptr, &err);
  require(buf3 && err == CL_SUCCESS, "clCreateBuffer in the second context");
  require(clReleaseMemObject(buf3) == CL_SUCCESS, "clReleaseMemObject in the second context");
  cl_context c3 = newContext(dev);
  _cl_context staleCopy = *c2;   // a context handle that outlives the context, as a stale cl_context would
  clReleaseContext(c2);
  size_t before = allocations.size();
  cl_mem stale = clCreateBuffer(&staleCopy, CL_MEM_READ_WRITE, 64, nullptr, &err);
  require(!stale && err != CL_SUCCESS, "clCreateBuffer for a destroyed context fails");
  require(allocations.size() == before, "nothing is allocated for a destroyed context");
  require(current() == c3->ctx, "a failed clCreateBuffer leaves the caller's context current");
  clReleaseContext(c3);

  if (failures) { std::printf("cuda release context test: %d FAILED\n", failures); return 1; }
  std::printf("cuda release context test PASS\n");
  return 0;
}
