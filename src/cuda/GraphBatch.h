#pragma once
// Include after tinycuda.h. This batches native launch calls without CUDA
// stream capture. Repetition is reserved for squares with stationary slots.
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class AevumCudaGraphBatch {
  struct Launch {
    CUfunction function{};
    unsigned blocks{}, threads{};
    int count{};
    alignas(16) std::array<char, _cl_kernel::MAX_ARG_BYTES> data{};
    std::array<size_t, _cl_kernel::MAX_ARGS> offsets{}, sizes{};
    bool sameShape(const Launch& other) const {
      return function == other.function && blocks == other.blocks &&
             threads == other.threads && count == other.count;
    }
    bool sameArguments(const Launch& other) const {
      if (count != other.count) return false;
      for (int i = 0; i < count; ++i)
        if (sizes[i] != other.sizes[i] ||
            std::memcmp(data.data() + offsets[i], other.data.data() + other.offsets[i], sizes[i]))
          return false;
      return true;
    }
    CUDA_KERNEL_NODE_PARAMS params(std::array<void*, _cl_kernel::MAX_ARGS>& pointers) {
      for (int i = 0; i < count; ++i) pointers[i] = data.data() + offsets[i];
      CUDA_KERNEL_NODE_PARAMS p{};
      p.func = function;
      p.gridDimX = blocks; p.gridDimY = p.gridDimZ = 1;
      p.blockDimX = threads; p.blockDimY = p.blockDimZ = 1;
      p.kernelParams = pointers.data();
      return p;
    }
  };
  struct Cached {
    CUgraph graph{};
    CUgraphExec exec{};
    std::vector<CUgraphNode> nodes;
    std::vector<Launch> shape;
    unsigned repeat{1};
    bool parametersValid{true};
    ~Cached() {
      if (exec) cuGraphExecDestroy(exec);
      if (graph) cuGraphDestroy(graph);
    }
  };
  bool active_{false};
  unsigned repeat_{1};
  std::vector<Launch> pending_;
  std::vector<std::unique_ptr<Cached>> cache_;
  static void check(CUresult r) {
    if (r != CUDA_SUCCESS)
      throw std::runtime_error("CUDA square graph API failure: " + std::to_string(int(r)));
  }
public:
  AevumCudaGraphBatch() { pending_.reserve(16); cache_.reserve(2); }
  bool active() const { return active_; }
  void begin(unsigned repeat = 1) {
    if (active_) throw std::runtime_error("nested CUDA square graph");
    if (!repeat || repeat > 128) throw std::runtime_error("CUDA square graph repeat must be in [1, 128]");
    pending_.clear();
    repeat_ = repeat;
    active_ = true;
  }
  void abort() noexcept { active_ = false; pending_.clear(); }
  void add(cl_kernel k, unsigned blocks, unsigned threads) {
    if (!active_ || pending_.size() >= 16 || !blocks || !threads ||
        k->numArgs < 0 || k->numArgs > _cl_kernel::MAX_ARGS)
      throw std::runtime_error("unsupported CUDA square graph launch");
    Launch launch;
    launch.function = k->func;
    launch.blocks = blocks;
    launch.threads = threads;
    launch.count = k->numArgs;
    std::memcpy(launch.data.data(), k->argData, launch.data.size());
    for (int i = 0; i < launch.count; ++i) {
      if (k->argOffsets[i] >= launch.data.size() ||
          k->argSizes[i] > launch.data.size() - k->argOffsets[i])
        throw std::runtime_error("invalid CUDA square graph argument span");
      launch.offsets[i] = k->argOffsets[i];
      launch.sizes[i] = k->argSizes[i];
    }
    pending_.push_back(launch); // Snapshot now: later setArg calls may reuse k.
  }
  void launch(CUstream stream) {
    if (!active_ || pending_.empty()) throw std::runtime_error("empty CUDA square graph");
    active_ = false;
    Cached* selected = nullptr;
    for (auto& cached : cache_) {
      if (cached->repeat != repeat_ || cached->shape.size() != pending_.size()) continue;
      bool same = true;
      for (size_t i = 0; i < pending_.size(); ++i)
        same = same && pending_[i].sameShape(cached->shape[i]);
      if (same) { selected = cached.get(); break; }
    }
    if (!selected) {
      if (cache_.size() >= 8) throw std::runtime_error("CUDA square graph topology limit");
      auto fresh = std::make_unique<Cached>();
      check(cuGraphCreate(&fresh->graph, 0));
      fresh->repeat = repeat_;
      for (unsigned r = 0; r < repeat_; ++r) {
        for (auto& item : pending_) {
          std::array<void*, _cl_kernel::MAX_ARGS> pointers{};
          auto p = item.params(pointers);
          CUgraphNode node{};
          const auto previous = fresh->nodes.empty() ? nullptr : &fresh->nodes.back();
          check(cuGraphAddKernelNode(&node, fresh->graph, previous, previous ? 1 : 0, &p));
          // Order the last node of each square before the first of the next.
          fresh->nodes.push_back(node);
        }
      }
#if CUDA_VERSION >= 11040
      check(cuGraphInstantiateWithFlags(&fresh->exec, fresh->graph, 0));
#else
      check(cuGraphInstantiate(&fresh->exec, fresh->graph, nullptr, nullptr, 0));
#endif
      fresh->shape = pending_;
      selected = fresh.get();
      cache_.push_back(std::move(fresh));
    } else {
      // Preserve per-square refresh for the original path. A repeated graph
      // can retain parameters only when the new snapshot is identical; changed
      // buffers or slots must reach every copy before it can be submitted.
      bool same = repeat_ > 1 && selected->parametersValid;
      for (size_t i = 0; same && i < pending_.size(); ++i)
        same = pending_[i].sameArguments(selected->shape[i]);
      if (!same) {
        // Invalidate first: a partial driver failure must force a full refresh
        // on retry, even when the caller returns to the previous argument set.
        selected->parametersValid = false;
        for (size_t i = 0; i < selected->nodes.size(); ++i) {
          std::array<void*, _cl_kernel::MAX_ARGS> pointers{};
          auto p = pending_[i % pending_.size()].params(pointers);
          check(cuGraphExecKernelNodeSetParams(selected->exec, selected->nodes[i], &p));
        }
        if (repeat_ > 1) selected->shape = pending_;
        selected->parametersValid = true;
      }
    }
    check(cuGraphLaunch(selected->exec, stream));
    pending_.clear();
  }
};
