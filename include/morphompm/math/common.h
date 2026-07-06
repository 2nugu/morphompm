#ifndef MORPHOMPM_COMMON_H
#define MORPHOMPM_COMMON_H

#include <cmath>
#include <algorithm>


#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

namespace morphompm {
namespace math {

// ============================================================
// Mathematical Constants
// ============================================================

constexpr float PI = 3.14159265359f;
constexpr float TWO_PI = 6.28318530718f;
constexpr float HALF_PI = 1.57079632679f;
constexpr float INV_PI = 0.31830988618f;
constexpr float EPSILON = 1e-6f;

constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// ============================================================
// Utility Functions
// ============================================================


/// Convert degrees to radians
HOST_DEVICE inline constexpr float deg_to_rad(float deg) {
    return deg * DEG_TO_RAD;
}

/// Convert radians to degrees
HOST_DEVICE inline constexpr float rad_to_deg(float rad) {
    return rad * RAD_TO_DEG;
}

/// Clamp value between min and max
template<typename T>
HOST_DEVICE inline constexpr T clamp(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : (value > max_val) ? max_val : value;
}

/// Linear interpolation
template<typename T>
HOST_DEVICE inline constexpr T lerp(T a, T b, float t) {
    return a * (1.0f - t) + b * t;
}

/// Check if value is approximately zero
/// Calculate absolute value in a constexpr context
HOST_DEVICE inline constexpr float constexpr_abs(float x) {
    return (x < 0.0f) ? -x : x;
}

/// Check if value is approximately zero
HOST_DEVICE inline constexpr bool is_zero(float value, float epsilon = EPSILON) {
    return constexpr_abs(value) < epsilon;
}

/// Check if two values are approximately equal
HOST_DEVICE inline constexpr bool approx_equal(float a, float b, float epsilon = EPSILON) {
    return constexpr_abs(a - b) < epsilon;
}

/// Square function
template<typename T>
HOST_DEVICE inline constexpr T square(T x) {
    return x * x;
}

/// Sign function (-1, 0, or 1)
template<typename T>
HOST_DEVICE inline constexpr int sign(T value) {
    return (T(0) < value) - (value < T(0));
}

/// Smooth step (Hermite interpolation)
HOST_DEVICE inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

/// Smoother step (Ken Perlin's version)
HOST_DEVICE inline float smootherstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

} // namespace math
} // namespace morphompm

#endif // MORPHOMPM_COMMON_H
