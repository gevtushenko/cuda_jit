//
// Created by egi on 1/18/20.
//

#include "cuda_jit.h"
#include "inja.hpp"

#include <nvrtc.h>
#include <cuda.h>

namespace cuda_jit
{

void throw_on_error (nvrtcResult result)
{
  if (result != NVRTC_SUCCESS)
    throw std::runtime_error (std::string ("NVRTC ERRROR: ") + nvrtcGetErrorString (result));
}

void throw_on_error (CUresult result)
{
  if (result != CUDA_SUCCESS)
    {
      const char *msg;
      cuGetErrorName (result, &msg);
      throw std::runtime_error (std::string ("CUDA ERRROR: ") + std::string (msg));
    }
}

class kernel_base::kernel_impl
{
public:
  CUmodule kernel_module;
  CUfunction kernel_fn;
};

kernel_base::kernel_base (kernel_base &&) = default;
kernel_base::kernel_base (const std::string &kernel_name, std::unique_ptr<char[]> ptx_arg)
  : ptx (std::move (ptx_arg))
  , impl (new kernel_base::kernel_impl ())
{
  throw_on_error (cuModuleLoadDataEx (&impl->kernel_module, ptx.get (), 0, nullptr, nullptr));
  throw_on_error (cuModuleGetFunction (&impl->kernel_fn, impl->kernel_module, kernel_name.c_str ()));
}

kernel_base::~kernel_base ()
{
  cuModuleUnload (impl->kernel_module);
}


std::string cuda_jit_base::gen_kernel (const nlohmann::json &json) const
{
  const std::string body = inja::render (body_template, json);
  return "extern \"C\" __global__ void " + name + " (" + params + ")" + body;
}

std::unique_ptr<char[]> cuda_jit_base::compile_base (const nlohmann::json &json)
{
  cudaFree (0);
  const std::string kernel_source = gen_kernel (json);

  nvrtcProgram prog;
  throw_on_error (nvrtcCreateProgram (&prog, kernel_source.c_str (), name.c_str (), 0, nullptr, nullptr));

  try
    {
      const char *options[] = { "--gpu-architecture=compute_75", "-fmad=true" };
      throw_on_error (nvrtcCompileProgram (prog, 2, options));
    }
  catch (...)
    {
      size_t log_size {};
      throw_on_error (nvrtcGetProgramLogSize (prog, &log_size));

      std::unique_ptr<char[]> log (new char[log_size]);
      throw_on_error (nvrtcGetProgramLog (prog, log.get ()));

      nvrtcDestroyProgram (&prog);
      throw std::runtime_error ("Compilation fail: " + std::string (log.get ()));
    }

  size_t ptx_size {};
  throw_on_error (nvrtcGetPTXSize (prog, &ptx_size));

  std::unique_ptr<char[]> ptx (new char[ptx_size]);
  throw_on_error (nvrtcGetPTX (prog, ptx.get ()));
  throw_on_error (nvrtcDestroyProgram (&prog));

  return std::move (ptx);
}

void kernel_base::launch_base (dim3 grid_size, dim3 block_size, void **params)
{
  throw_on_error (cuLaunchKernel (impl->kernel_fn, grid_size.x, grid_size.y, grid_size.z, block_size.x, block_size.y, block_size.z, 0, 0, params, nullptr));
}

}
