//
// Created by egi on 1/18/20.
//

#include <cuda_jit.h>

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
    }, (float, a), (const float *, x), (const float *, y), (float *, out));

  nlohmann::json data;
  data["N"] = 1024;
  saxpy.render (data);
  std::cout << saxpy.get_body () << std::endl;
}