//
// Created by egi on 1/18/20.
//

#include "cuda_jit.h"
#include "inja.hpp"


void cuda_jit_base::render (const nlohmann::json &json)
{
  body = inja::render (body_template, json);
}
