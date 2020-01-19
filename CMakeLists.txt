cmake_minimum_required(VERSION 3.15)
project(cuda_jit)

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA REQUIRED)
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

add_library(cuda_jit)
target_sources(cuda_jit PUBLIC include/cuda_jit.h include/map.h PRIVATE src/cuda_jit.cpp)
target_include_directories(cuda_jit PUBLIC include external/json PRIVATE external/inja)
target_link_libraries(cuda_jit PRIVATE ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} ${CUDA_CUDA_LIBRARY})

add_executable(direct_jit main.cpp)
target_include_directories(direct_jit PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(direct_jit ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} ${CUDA_CUDA_LIBRARY})

add_executable(example)
target_sources(example PRIVATE example.cpp)
target_link_libraries(example cuda_jit)