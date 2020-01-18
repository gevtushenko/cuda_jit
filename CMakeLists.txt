cmake_minimum_required(VERSION 3.15)
project(cuda_jit)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)
find_library(CUDA_NVRTC_LIB libnvrtc nvrtc HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${LIBNVRTC_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64" /usr/lib64 /usr/local/cuda/lib64)

add_executable(cuda_jit main.cpp)
target_include_directories(cuda_jit PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(cuda_jit ${CUDA_LIBRARIES} ${CUDA_NVRTC_LIB} ${CUDA_CUDA_LIBRARY})