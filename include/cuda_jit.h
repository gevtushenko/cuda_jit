//
// Created by egi on 1/18/20.
//

#ifndef CUDA_JIT_H
#define CUDA_JIT_H

#include <iostream>
#include <string>

#include "map.h"

template <typename... args_types>
class cuda_jit
{
  const std::string name;
  const std::string params;
  const std::string body;
public:
  cuda_jit (
    const std::string &kernel_name,
    const std::string &kernel_params,
    const std::string &kernel_body)
    : name (kernel_name)
    , params (kernel_params)
    , body (kernel_body)
  {
    std::cout << "__global__ void " << name << " (" << params << ") \n" << body << std::endl;
  }

  void operator ()(args_types... args)
  {

  }
};

#define GET_FIRST(f, s) f
#define APPLY_TO_PAIR(function, pair) function pair
#define GET_FIRST_FROM_PAIR(pair) APPLY_TO_PAIR(GET_FIRST, pair)

#define CONCAT_PAIR_IMPL(f, s) f s
#define CONCAT_PAIR(pair) CONCAT_PAIR_IMPL pair

#define DEFER_STRINGIFY(args...) STRINGIFY(args)
#define STRINGIFY(args...) #args

#define jit(name, body, args...) cuda_jit<MAP_LIST(GET_FIRST_FROM_PAIR, args)> name (#name, DEFER_STRINGIFY(MAP_LIST(CONCAT_PAIR, args)), #body)

#endif // CUDA_JIT_H
