// Host control gate for the actual graph batching header, with a recording
// driver facade. This does not compile CUDA kernels, run a graph or emulate GPU
// arithmetic/order. Hardware gate remains mandatory.
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#define CUDA_VERSION 13000
using CUresult = int;
constexpr int CUDA_SUCCESS = 0;
using CUfunction = uintptr_t;
using CUstream = int;
struct Params;
struct Node { CUfunction f; unsigned blocks, threads; std::array<uint64_t,2> args; };
struct Graph { std::vector<Node*> nodes; };
struct Exec { std::vector<Node> nodes; std::vector<Node*> identity; };
using CUgraphNode = Node*;
using CUgraph = Graph*;
using CUgraphExec = Exec*;
struct CUDA_KERNEL_NODE_PARAMS {
  CUfunction func{};
  unsigned gridDimX{}, gridDimY{}, gridDimZ{};
  unsigned blockDimX{}, blockDimY{}, blockDimZ{}, sharedMemBytes{};
  void** kernelParams{};
  void** extra{};
};
struct _cl_kernel {
  static constexpr int MAX_ARGS=32, MAX_ARG_BYTES=512;
  CUfunction func{};
  int numArgs=2;
  char argData[MAX_ARG_BYTES]{};
  size_t argOffsets[MAX_ARGS]{},argSizes[MAX_ARGS]{};
};
using cl_kernel = _cl_kernel*;
static int creates=0, updates=0, launches=0;
static bool rejectUpdate=false;
static int updatesBeforeFailure=-1;
static std::vector<std::vector<Node>> submitted;
Node copyParams(const CUDA_KERNEL_NODE_PARAMS* p) {
  assert(p->gridDimY==1 && p->gridDimZ==1 && p->blockDimY==1 && p->blockDimZ==1);
  Node n{p->func,p->gridDimX,p->blockDimX,{}};
  for(int i=0;i<2;++i) std::memcpy(&n.args[i],p->kernelParams[i],8);
  return n;
}
CUresult cuGraphCreate(CUgraph* out,unsigned) { *out=new Graph; ++creates; return 0; }
CUresult cuGraphAddKernelNode(CUgraphNode* out,CUgraph g,const CUgraphNode* deps,size_t count,const CUDA_KERNEL_NODE_PARAMS* p) {
  assert(count==(g->nodes.empty()?0u:1u));
  if(count) assert(*deps==g->nodes.back());
  *out=new Node(copyParams(p)); g->nodes.push_back(*out); return 0;
}
CUresult cuGraphInstantiateWithFlags(CUgraphExec* out,CUgraph g,unsigned long long) {
  *out=new Exec;
  for(auto* n:g->nodes) { (*out)->nodes.push_back(*n); (*out)->identity.push_back(n); }
  return 0;
}
CUresult cuGraphExecKernelNodeSetParams(CUgraphExec e,CUgraphNode n,const CUDA_KERNEL_NODE_PARAMS* p) {
  if(rejectUpdate || updatesBeforeFailure==0) return 1;
  if(updatesBeforeFailure>0) --updatesBeforeFailure;
  for(size_t i=0;i<e->nodes.size();++i)
    if(e->identity[i]==n) { e->nodes[i]=copyParams(p); ++updates; return 0; }
  return 1;
}
CUresult cuGraphLaunch(CUgraphExec e,CUstream) { submitted.push_back(e->nodes); ++launches; return 0; }
CUresult cuGraphExecDestroy(CUgraphExec e) { delete e; return 0; }
CUresult cuGraphDestroy(CUgraph g) { for(auto* n:g->nodes)delete n; delete g; return 0; }
#include "../src/cuda/GraphBatch.h"
void set(_cl_kernel& k, uint64_t buffer, uint64_t slot) {
  k.argOffsets[0]=0; k.argOffsets[1]=8;
  k.argSizes[0]=k.argSizes[1]=8;
  std::memcpy(k.argData,&buffer,8); std::memcpy(k.argData+8,&slot,8);
}
int main() {
  AevumCudaGraphBatch b(false);
  _cl_kernel middle,carry;
  middle.func=1; carry.func=2;
  b.begin();
  set(middle,100,0); b.add(&middle,4096,64);
  set(carry,100,17); b.add(&carry,1024,256);
  set(middle,999,999); set(carry,999,999); // Snapshot must outlive reuse.
  b.launch(0);
  assert(creates==1 && launches==1 && updates==0);
  assert(submitted[0][0].args[0]==100 && submitted[0][1].args[1]==17);
  b.begin();
  set(middle,200,1); b.add(&middle,4096,64);
  set(carry,200,18); b.add(&carry,1024,256);
  b.launch(0);
  assert(creates==1 && updates==2 && launches==2);
  assert(submitted[1][0].args[0]==200 && submitted[1][1].args[1]==18);
  assert(submitted[0][1].args[1]==17); // Earlier launch remains unchanged.
  b.begin(); b.add(&middle,4096,64); b.abort();
  assert(!b.active() && launches==2);
  b.begin(); carry.func=3; b.add(&carry,1024,256); b.launch(0);
  assert(creates==2 && launches==3); // ROE/non-ROE function changes topology.
  b.begin(); b.add(&middle,4096,64); carry.func=2; b.add(&carry,1024,256);
  rejectUpdate=true;
  bool rejected=false;
  try { b.launch(0); } catch(const std::runtime_error&) { rejected=true; b.abort(); }
  assert(rejected && launches==3); // Never submit stale parameters on error.
  rejectUpdate=false;
  b.begin(); middle.argOffsets[0]=512;
  rejected=false;
  try { b.add(&middle,1,64); } catch(const std::runtime_error&) { rejected=true; b.abort(); }
  assert(rejected);
  // The same kernel handle can be configured twice before graph submission.
  middle.argOffsets[0]=0;
  for(int repeat=0;repeat<2;++repeat) {
    b.begin();
    set(middle,300+repeat,30+repeat); b.add(&middle,4096,64);
    set(middle,400+repeat,40+repeat); b.add(&middle,4096,64);
    b.launch(0);
    const auto& last=submitted.back();
    assert(last[0].args[0]==uint64_t(300+repeat));
    assert(last[1].args[0]==uint64_t(400+repeat));
    assert(last[0].args[1]==uint64_t(30+repeat));
    assert(last[1].args[1]==uint64_t(40+repeat));
  }
  assert(creates==3 && launches==5);
  // Unchanged values must still refresh every node in the default-OFF mode.
  for (bool reuse : {false, true}) {
    AevumCudaGraphBatch cached(reuse);
    _cl_kernel a, z;
    a.func=10; z.func=11;
    set(a,1000,70); set(z,2000,80);
    auto submit = [&] {
      cached.begin(); cached.add(&a,4096,64); cached.add(&z,1024,256);
      cached.launch(0);
    };
    submit();
    int before=updates;
    submit();
    assert(updates-before==(reuse ? 0 : 2));
    // One live argument changes: only that node can be reused selectively.
    set(z,2000,81);
    before=updates; submit();
    assert(updates-before==(reuse ? 1 : 2));
    assert(submitted.back()[0].args[0]==1000 && submitted.back()[1].args[1]==81);
    // Byte-identical values packed elsewhere remain identical parameters.
    std::memmove(a.argData+32,a.argData,16);
    a.argOffsets[0]=32; a.argOffsets[1]=40;
    a.argData[511]=42; // Not a live argument; do not compare padding.
    before=updates; submit();
    assert(updates-before==(reuse ? 0 : 2));
    // A size change must never be treated as an unchanged argument span.
    a.argSizes[0]=4;
    before=updates; submit();
    assert(updates-before==(reuse ? 1 : 2));
    a.argSizes[0]=8;
    before=updates; submit();
    assert(updates-before==(reuse ? 1 : 2));

    // A partially successful refresh must not poison the executable snapshot.
    set(a,1001,71); set(z,2001,82);
    const int beforeLaunch=launches;
    before=updates; updatesBeforeFailure=1;
    bool failed=false;
    try { submit(); } catch(const std::runtime_error&) { failed=true; cached.abort(); }
    assert(failed && launches==beforeLaunch && updates==before+1);
    updatesBeforeFailure=-1;
    before=updates; submit();
    assert(updates-before==(reuse ? 1 : 2));
    assert(submitted.back()[0].args[0]==1001 && submitted.back()[1].args[0]==2001);
    assert(submitted.back()[0].args[1]==71 && submitted.back()[1].args[1]==82);

    // Switching topology and returning must compare against that executable.
    z.func=12; submit();
    z.func=11; before=updates; submit();
    assert(updates-before==(reuse ? 0 : 2));
    assert(submitted.back()[1].f==11);
  }

  // Repeated handles represent distinct nodes with distinct snapshots.
  {
    AevumCudaGraphBatch cached(true);
    _cl_kernel k; k.func=20;
    auto submit = [&](uint64_t second) {
      cached.begin();
      set(k,3000,90); cached.add(&k,4096,64);
      set(k,second,91); cached.add(&k,4096,64);
      set(k,9999,9999);
      cached.launch(0);
    };
    submit(4000);
    int before=updates; submit(4000); assert(updates==before);
    submit(4001); assert(updates==before+1);
    assert(submitted.back()[0].args[0]==3000 && submitted.back()[1].args[0]==4001);
    assert(submitted.back()[0].args[1]==90 && submitted.back()[1].args[1]==91);
  }

  // Strict environment parsing and default-constructor behavior.
  const char* oldEnv=std::getenv("AEVUM_CUDA_GRAPH_REUSE_PARAMS");
  const bool hadEnv=oldEnv!=nullptr;
  const std::string savedEnv=oldEnv ? oldEnv : "";
  auto environment = [](const char* value) {
#ifdef _WIN32
    _putenv_s("AEVUM_CUDA_GRAPH_REUSE_PARAMS",value ? value : "");
#else
    if(value) setenv("AEVUM_CUDA_GRAPH_REUSE_PARAMS",value,1);
    else unsetenv("AEVUM_CUDA_GRAPH_REUSE_PARAMS");
#endif
  };
  for (const char* value : {static_cast<const char*>(nullptr), "0", "1"}) {
    environment(value);
    AevumCudaGraphBatch cached;
    _cl_kernel k; k.func=30; set(k,5000,100);
    cached.begin(); cached.add(&k,1,64); cached.launch(0);
    int before=updates;
    cached.begin(); cached.add(&k,1,64); cached.launch(0);
    assert(updates-before==((value && std::strcmp(value,"1")==0) ? 0 : 1));
  }
  for (const char* value : {"2", "true", "01"}) {
    environment(value);
    bool failed=false;
    try { AevumCudaGraphBatch invalid; } catch(const std::runtime_error&) { failed=true; }
    assert(failed);
  }
  environment(hadEnv ? savedEnv.c_str() : nullptr);
  std::cout<<"graph snapshots, selective updates, partial failure and option gates checked\n";
}
