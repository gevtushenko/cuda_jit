//
// Created by egi on 1/18/20.
//

#ifndef CUDA_JIT_H
#define CUDA_JIT_H

#include <iostream>
#include <memory>
#include <string>

#include "nlohmann/json.hpp"

#include <cuda_runtime_api.h> /// For dim3

#include "map.h"

namespace cuda_jit
{

class kernel_base
{
  class kernel_impl;

public:
  kernel_base (kernel_base &&);
  kernel_base (const std::string &kernel_name, std::unique_ptr<char[]> ptx_arg);
  ~kernel_base ();

protected:
  void launch_base (dim3 grid_size, dim3 block_size, void **params);

private:
  std::unique_ptr<char[]> ptx;
  std::unique_ptr<kernel_impl> impl;
};

template<typename... args_types>
class kernel : public kernel_base
{
  std::vector<void*> params;

public:
  kernel (kernel &&) = default;
  explicit kernel (const std::string &kernel_name, std::unique_ptr<char[]> ptx_arg)
    : kernel_base (kernel_name, std::move (ptx_arg))
  { }

  void launch (dim3 grid_size, dim3 block_size, args_types... args)
  {
    params = { &args... };
    kernel_base::launch_base (grid_size, block_size, params.data ());
  }
};

class cuda_jit_base
{
protected:
  const std::string name;
  const std::string params;
  const std::string body_template;

public:
  cuda_jit_base (
    const std::string &kernel_name,
    const std::string &kernel_params,
    const std::string &kernel_body)
    : name (kernel_name), params (kernel_params), body_template (kernel_body)
  {
  }

protected:
  std::unique_ptr<char[]> compile_base (const nlohmann::json &json);

private:
  std::string gen_kernel (const nlohmann::json &json) const;
};

template<typename... args_types>
class cuda_jit : public cuda_jit_base
{
public:
  cuda_jit (
    const std::string &kernel_name,
    const std::string &kernel_params,
    const std::string &kernel_body)
    : cuda_jit_base (kernel_name, kernel_params, kernel_body)
  {}

  kernel<args_types...> compile (const nlohmann::json &json)
  {
    return kernel<args_types...> (name, cuda_jit_base::compile_base (json));
  }
};

}

#define GET_FIRST(f, s) f
#define APPLY_TO_PAIR(function, pair) function pair
#define GET_FIRST_FROM_PAIR(pair) APPLY_TO_PAIR(GET_FIRST, pair)

#define CONCAT_PAIR_IMPL(f, s) f s
#define CONCAT_PAIR(pair) CONCAT_PAIR_IMPL pair

#define DEFER_STRINGIFY(args...) STRINGIFY(args)
#define STRINGIFY(args...) #args

#define jit(name, body, args...) cuda_jit::cuda_jit<MAP_LIST(GET_FIRST_FROM_PAIR, args)> name (#name, DEFER_STRINGIFY(MAP_LIST(CONCAT_PAIR, args)), #body)

#endif // CUDA_JIT_H
