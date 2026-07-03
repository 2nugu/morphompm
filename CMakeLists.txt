cmake_minimum_required(VERSION 3.16)
project(morphompm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Self-contained: math core (Vec3 / Matrix3 / SVD) vendored under
# include/basements/core/math/ (originally from Basements; no external path).
add_executable(test_growth tests/test_growth.cpp)
target_include_directories(test_growth PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Vec3 uses AVX2 intrinsics (immintrin.h).
if (MSVC)
    target_compile_options(test_growth PRIVATE /arch:AVX2 /O2)
else()
    target_compile_options(test_growth PRIVATE -mavx2 -mfma -O2)
endif()
