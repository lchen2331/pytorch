#pragma once

#include <string>
#include <vector>

// Shared struct for Triton kernel compilation results.
// Used by both cpp-wrapper JIT (filled at runtime via Python) and AOTInductor
// (filled at compile time from a generated config header).
struct LazyKernelCompileResult {
  std::string cubin_path;
  std::string mangled_name;
  int num_warps;
  int shared_mem;
  // Runtime-only launch metadata used by the JIT C++ wrapper. AOTI native
  // launchers keep these fields at their default values and do not consume
  // them.
  bool launch_cooperative_grid;
  int max_cooperative_groups;
  std::vector<int> xblocks;
  std::vector<int> yblocks;
  std::vector<int> zblocks;
  std::vector<int> r0blocks;
  int rsplit;
  int rsplit_size;
  int config_index;
  int global_scratch;
  int profile_scratch;
};
