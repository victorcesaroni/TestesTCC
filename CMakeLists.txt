cmake_minimum_required(VERSION 3.10.2)

find_package(CUDA REQUIRED QUIET)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_61,code=[sm_61,compute_61]; # GTX 1050 T
    --expt-extended-lambda;
    --std=c++11;
    -Xcompiler -fopenmp --compiler-options -fPIC -O3 -D__INTEL_COMPILER
    )

include_directories(
    src
    )

cuda_add_executable(
    tccbin
    src/common.h
    src/common.cu
    src/memory_manager.h
    src/misc.h
    src/misc.cu
    src/filters.h
    src/filters.cu
    src/cuda_timer.h
    src/cuda_timer.cu
    src/matrix_accessor.h
    src/main.cu
    )

target_link_libraries(
    tccbin
    )

