#include <iostream>
#include <memory>

#include <nvrtc.h>
#include <cuda.h>

void throw_on_error (nvrtcResult result, const char *file, const int line)
{
  if (result != NVRTC_SUCCESS)
    throw std::runtime_error (std::string ("NVRTC ERRROR: ") + nvrtcGetErrorString (result));
}

void throw_on_error (CUresult result, const char *file, const int line)
{
  if (result != CUDA_SUCCESS)
    {
      const char *msg;
      cuGetErrorName (result, &msg);
      throw std::runtime_error (std::string ("CUDA ERRROR: ") + std::string (msg));
    }
}

int main ()
{
  const char *saxpy = "\
extern \"C\" __global__ void saxpy (float a, const float *x, const float *y, float *out, size_t n) \n\
{                                                                                                  \n\
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;                                        \n\
  if (tid < n)                                                                                     \n\
    out[tid] = a * x[tid] + y[tid];                                                                \n\
}                                                                                                  \n";

  std::cout << "SRC: \n" << saxpy << std::endl;

  nvrtcProgram prog;
  throw_on_error (nvrtcCreateProgram (&prog, saxpy, "saxpy.cu", 0, nullptr, nullptr), __FILE__, __LINE__);

  try
    {
      const char *options[] = { "--gpu-architecture=compute_75", "-fmad=true" };
      throw_on_error (nvrtcCompileProgram (prog, 2, options), __FILE__, __LINE__);
    }
  catch (...)
    {
      size_t log_size {};
      throw_on_error (nvrtcGetProgramLogSize (prog, &log_size), __FILE__, __LINE__);

      std::unique_ptr<char[]> log (new char[log_size]);
      throw_on_error (nvrtcGetProgramLog (prog, log.get ()), __FILE__, __LINE__);

      std::cerr << "LOG: \n" << log.get () << std::endl;
      throw_on_error (nvrtcDestroyProgram (&prog), __FILE__, __LINE__);
      return 1;
    }

  size_t ptx_size {};
  throw_on_error (nvrtcGetPTXSize (prog, &ptx_size), __FILE__, __LINE__);

  std::unique_ptr<char[]> ptx (new char[ptx_size]);
  throw_on_error (nvrtcGetPTX (prog, ptx.get ()), __FILE__, __LINE__);
  throw_on_error (nvrtcDestroyProgram (&prog), __FILE__, __LINE__);

  std::cout << "PTX:\n" << ptx.get () << std::endl;

  CUdevice device;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

  throw_on_error (cuInit (0), __FILE__, __LINE__);
  throw_on_error (cuDeviceGet (&device, 0), __FILE__, __LINE__);
  throw_on_error (cuCtxCreate (&context, 0, device), __FILE__, __LINE__);
  throw_on_error (cuModuleLoadDataEx (&module, ptx.get (), 0, nullptr, nullptr), __FILE__, __LINE__);
  throw_on_error (cuModuleGetFunction (&kernel, module, "saxpy"), __FILE__, __LINE__);

  size_t n_blocks = 32;
  size_t n_threads = 1024;
  size_t n = n_blocks * n_threads;
  size_t buffer_size = n * sizeof (float);

  std::unique_ptr<float[]> h_x (new float[n]);
  std::unique_ptr<float[]> h_y (new float[n]);
  std::unique_ptr<float[]> h_out (new float[n]);

  for (size_t i = 0; i < n; i++)
    {
      h_x[i] = static_cast<float> (i);
      h_y[i] = static_cast<float> (i + i);
    }

  CUdeviceptr d_x, d_y, d_out;
  throw_on_error(cuMemAlloc (&d_x, buffer_size), __FILE__, __LINE__);
  throw_on_error(cuMemAlloc (&d_y, buffer_size), __FILE__, __LINE__);
  throw_on_error(cuMemAlloc (&d_out, buffer_size), __FILE__, __LINE__);
  throw_on_error(cuMemcpyHtoD (d_x, h_x.get (), buffer_size), __FILE__, __LINE__);
  throw_on_error(cuMemcpyHtoD (d_y, h_y.get (), buffer_size), __FILE__, __LINE__);

  float a = 42.0f;
  void *kernel_args[] = { &a, &d_x, &d_y, &d_out, &n };
  throw_on_error (
    cuLaunchKernel (kernel, n_blocks, 1, 1, n_threads, 1, 1, 0, 0, kernel_args, nullptr),
    __FILE__, __LINE__);
  throw_on_error (cuCtxSynchronize (), __FILE__, __LINE__);

  throw_on_error(cuMemcpyDtoH (h_out.get (), d_out, buffer_size), __FILE__, __LINE__);

  #if 0
  for (size_t i = 0; i < n; i++)
    std::cout << "out[" << i << "] = " << h_out[i] << "\n";
  #endif

  throw_on_error (cuMemFree (d_x), __FILE__, __LINE__);
  throw_on_error (cuMemFree (d_y), __FILE__, __LINE__);
  throw_on_error (cuMemFree (d_out), __FILE__, __LINE__);

  throw_on_error (cuModuleUnload (module), __FILE__, __LINE__);
  throw_on_error (cuCtxDestroy (context), __FILE__, __LINE__);


  return 0;
}
