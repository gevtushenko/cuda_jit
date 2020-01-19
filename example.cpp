//
// Created by egi on 1/18/20.
//

#include <cuda_jit.h>
#include <cuda_runtime.h>

#include "map.h"

int main ()
{
  jit (saxpy,
    {
      const size_t tid = threadIdx.x;

      if (tid < {{ N }})
        {
          // Test comment
          out[tid] = a * x[tid] + y[tid];
        }
    }, (int, a), (const int *, x), (const int *, y), (int *, out));

  size_t n = 1024;

  nlohmann::json data;
  data["N"] = n;

  int a = 2;
  int *x {};
  int *y {};
  int *out {};

  cudaMalloc (&x, n * sizeof (int));
  cudaMalloc (&y, n * sizeof (int));
  cudaMalloc (&out, n * sizeof (int));

  std::unique_ptr<int[]> h_x (new int[n]);
  std::unique_ptr<int[]> h_y (new int[n]);
  std::unique_ptr<int[]> h_out (new int[n]);

  for (size_t i = 0; i < n; i++)
    {
      h_x[i] = 1;
      h_y[i] = 2;
    }

  cudaMemcpy (x, h_x.get (), n * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpy (y, h_y.get (), n * sizeof (float), cudaMemcpyHostToDevice);

  auto saxpy_kernel = saxpy.compile (data);
  saxpy_kernel.launch (1, 1024, a, x, y, out);

  cudaMemcpy (h_out.get (), out, n * sizeof (float), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < n; i++)
    {
      int target_value = a * h_x[i] + h_y[i];
      if (target_value != h_out[i])
        std::cerr << "Error in out[" << i << "] = " << h_out[i] << " != " << target_value << "\n";
    }

  cudaFree (x);
  cudaFree (y);
  cudaFree (out);
}