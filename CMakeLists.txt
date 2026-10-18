cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(kpm_tightbinding LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen is allowed ONLY as the R1 exact-diagonalization oracle
# (exact_diag, test_kpm). The KPM path itself is dependency-free.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(kpm_core src/lattice.cpp src/kpm.cpp)
target_include_directories(kpm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(kpm_core PRIVATE -Wall -Wextra)

# Hand-written R3/R4 kernels; no vendor library kernels in the measured path.
add_library(kpm_gpu STATIC src/kpm_gpu.cu)
target_include_directories(kpm_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kpm_gpu PUBLIC kpm_core CUDA::cudart)
target_compile_definitions(kpm_gpu PRIVATE KPM_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(exact_diag src/exact_diag.cpp)
target_link_libraries(exact_diag PRIVATE kpm_core Eigen3::Eigen)
target_compile_options(exact_diag PRIVATE -Wall -Wextra)
target_compile_definitions(exact_diag PRIVATE
  KPM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(dos_kpm src/dos_kpm.cpp)
target_link_libraries(dos_kpm PRIVATE kpm_core)
target_compile_options(dos_kpm PRIVATE -Wall -Wextra)
target_compile_definitions(dos_kpm PRIVATE
  KPM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(bench_kpm benchmarks/bench_kpm.cpp)
target_link_libraries(bench_kpm PRIVATE kpm_gpu)
target_compile_options(bench_kpm PRIVATE -Wall -Wextra)
target_compile_definitions(bench_kpm PRIVATE
  KPM_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(test_kpm tests/test_kpm.cpp)
target_link_libraries(test_kpm PRIVATE kpm_core Eigen3::Eigen)
target_compile_options(test_kpm PRIVATE -Wall -Wextra)
add_test(NAME cpu COMMAND test_kpm)

add_executable(test_gpu_kpm tests/test_gpu_kpm.cpp)
target_link_libraries(test_gpu_kpm PRIVATE kpm_gpu)
target_compile_options(test_gpu_kpm PRIVATE -Wall -Wextra)
add_test(NAME gpu COMMAND test_gpu_kpm)
