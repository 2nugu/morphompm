#ifndef BASEMENTS_VEC3_H
#define BASEMENTS_VEC3_H

#include <cmath>
#include <algorithm>
#include <immintrin.h>  // AVX/AVX2 intrinsics
#include "basements/core/math/common.h"

namespace basements {
namespace math {

// Epsilon is defined in common.h


/**
 * @brief 3D vector with SIMD optimization (AVX2)
 * 
 * Memory layout: 16-byte aligned for SIMD operations
 * Components: x, y, z, w (w is padding for alignment)
 */
struct alignas(16) Vec3 {
    float x, y, z, w;  // w is padding for SIMD (unused)
    
    // ============================================================
    // Constructors
    // ============================================================
    
    /// Default constructor: zero vector
    HOST_DEVICE Vec3() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    
    /// Parameterized constructor
    HOST_DEVICE Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_), w(0.0f) {}
    
    /// Scalar constructor: all components equal
    HOST_DEVICE explicit Vec3(float scalar) : x(scalar), y(scalar), z(scalar), w(0.0f) {}
    
    // ============================================================
    // Arithmetic Operators
    // ============================================================
    
    HOST_DEVICE Vec3 operator+(const Vec3& other) const {
        #if defined(__AVX__) && !defined(__CUDA_ARCH__)
        __m128 a = _mm_load_ps(&x);
        __m128 b = _mm_load_ps(&other.x);
        __m128 result = _mm_add_ps(a, b);
        Vec3 v;
        _mm_store_ps(&v.x, result);
        return v;
        #else
        return Vec3(x + other.x, y + other.y, z + other.z);
        #endif
    }
    
    HOST_DEVICE Vec3 operator-(const Vec3& other) const {
        #if defined(__AVX__) && !defined(__CUDA_ARCH__)
        __m128 a = _mm_load_ps(&x);
        __m128 b = _mm_load_ps(&other.x);
        __m128 result = _mm_sub_ps(a, b);
        Vec3 v;
        _mm_store_ps(&v.x, result);
        return v;
        #else
        return Vec3(x - other.x, y - other.y, z - other.z);
        #endif
    }
    
    HOST_DEVICE Vec3 operator*(float scalar) const {
        #if defined(__AVX__) && !defined(__CUDA_ARCH__)
        __m128 a = _mm_load_ps(&x);
        __m128 s = _mm_set1_ps(scalar);
        __m128 result = _mm_mul_ps(a, s);
        Vec3 v;
        _mm_store_ps(&v.x, result);
        return v;
        #else
        return Vec3(x * scalar, y * scalar, z * scalar);
        #endif
    }
    
    HOST_DEVICE Vec3 operator/(float scalar) const {
        // Avoid division by zero
        #ifdef __CUDA_ARCH__
        if (fabsf(scalar) < EPSILON) {
        #else
        if (std::abs(scalar) < EPSILON) {
        #endif
            return Vec3(0.0f);
        }
        
        #if defined(__AVX__) && !defined(__CUDA_ARCH__)
        __m128 a = _mm_load_ps(&x);
        __m128 s = _mm_set1_ps(scalar);
        __m128 result = _mm_div_ps(a, s);
        Vec3 v;
        _mm_store_ps(&v.x, result);
        return v;
        #else
        float inv = 1.0f / scalar;
        return Vec3(x * inv, y * inv, z * inv);
        #endif
    }
    
    HOST_DEVICE Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }
    
    HOST_DEVICE bool operator==(const Vec3& other) const {
        return (fabsf(x - other.x) < EPSILON) && 
               (fabsf(y - other.y) < EPSILON) &&
               (fabsf(z - other.z) < EPSILON);
    }

    HOST_DEVICE bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }
    
    // Compound assignment operators
    HOST_DEVICE Vec3& operator+=(const Vec3& other) {
        *this = *this + other;
        return *this;
    }
    
    HOST_DEVICE Vec3& operator-=(const Vec3& other) {
        *this = *this - other;
        return *this;
    }
    
    HOST_DEVICE Vec3& operator*=(float scalar) {
        *this = *this * scalar;
        return *this;
    }
    
    HOST_DEVICE Vec3& operator/=(float scalar) {
        *this = *this / scalar;
        return *this;
    }
    
    // ============================================================
    // Dot Product (SIMD optimized)
    // ============================================================
    
    HOST_DEVICE float dot(const Vec3& other) const {
        #if defined(__AVX__) && !defined(__CUDA_ARCH__)
        __m128 a = _mm_load_ps(&x);
        __m128 b = _mm_load_ps(&other.x);
        __m128 mul = _mm_mul_ps(a, b);
        
        // Horizontal add: sum all components
        __m128 sum = _mm_hadd_ps(mul, mul);
        sum = _mm_hadd_ps(sum, sum);
        
        return _mm_cvtss_f32(sum);
        #else
        return x * other.x + y * other.y + z * other.z;
        #endif
    }
    
    // ============================================================
    // Cross Product
    // ============================================================
    
    HOST_DEVICE Vec3 cross(const Vec3& other) const {
        return Vec3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    // ============================================================
    // Length and Normalization
    // ============================================================
    
    HOST_DEVICE float length_squared() const {
        return dot(*this);
    }
    
    HOST_DEVICE float length() const {
        #ifdef __CUDA_ARCH__
        return sqrtf(length_squared());
        #else
        return std::sqrt(length_squared());
        #endif
    }
    
    HOST_DEVICE Vec3 normalized() const {
        float len = length();
        if (len < EPSILON) {
            return Vec3(0.0f);  // Return zero vector for zero-length
        }
        return *this / len;
    }
    
    HOST_DEVICE void normalize() {
        *this = normalized();
    }
    
    // ============================================================
    // Distance
    // ============================================================
    
    HOST_DEVICE float distance(const Vec3& other) const {
        return (*this - other).length();
    }
    
    HOST_DEVICE float distance_squared(const Vec3& other) const {
        return (*this - other).length_squared();
    }
    
    // ============================================================
    // Utility Functions
    // ============================================================
    
    HOST_DEVICE bool is_zero(float epsilon = EPSILON) const {
        return length_squared() < epsilon * epsilon;
    }
    
    HOST_DEVICE bool approx_equal(const Vec3& other, float epsilon = EPSILON) const {
        return (*this - other).is_zero(epsilon);
    }
    
    // Component access
    HOST_DEVICE float& operator[](int index) {
        return (&x)[index];
    }
    
    HOST_DEVICE const float& operator[](int index) const {
        return (&x)[index];
    }
    
    // ============================================================
    // Static Factory Methods
    // ============================================================
    
    HOST_DEVICE static Vec3 zero() { return Vec3(0.0f, 0.0f, 0.0f); }
    HOST_DEVICE static Vec3 one() { return Vec3(1.0f, 1.0f, 1.0f); }
    HOST_DEVICE static Vec3 unit_x() { return Vec3(1.0f, 0.0f, 0.0f); }
    HOST_DEVICE static Vec3 unit_y() { return Vec3(0.0f, 1.0f, 0.0f); }
    HOST_DEVICE static Vec3 unit_z() { return Vec3(0.0f, 0.0f, 1.0f); }
    
    // ============================================================
    // Static Utility Functions
    // ============================================================
    
    HOST_DEVICE static Vec3 min(const Vec3& a, const Vec3& b) {
        #ifdef __CUDA_ARCH__
        return Vec3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
        #else
        return Vec3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z));
        #endif
    }
    
    HOST_DEVICE static Vec3 max(const Vec3& a, const Vec3& b) {
        #ifdef __CUDA_ARCH__
        return Vec3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
        #else
        return Vec3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z));
        #endif
    }
    
    HOST_DEVICE static Vec3 lerp(const Vec3& a, const Vec3& b, float t) {
        return a * (1.0f - t) + b * t;
    }
    
    HOST_DEVICE static Vec3 clamp(const Vec3& v, const Vec3& min_val, const Vec3& max_val) {
        return max(min_val, min(v, max_val));
    }
    
    // Reflect vector v around normal n
    HOST_DEVICE static Vec3 reflect(const Vec3& v, const Vec3& n) {
        return v - n * (2.0f * v.dot(n));
    }
    
    // Project vector v onto vector onto
    HOST_DEVICE static Vec3 project(const Vec3& v, const Vec3& onto) {
        float onto_len_sq = onto.length_squared();
        if (onto_len_sq < EPSILON) {
            return Vec3::zero();
        }
        return onto * (v.dot(onto) / onto_len_sq);
    }
};

// ============================================================
// Non-member operators
// ============================================================

HOST_DEVICE inline Vec3 operator*(float scalar, const Vec3& v) {
    return v * scalar;
}

} // namespace math
} // namespace basements

#endif // BASEMENTS_VEC3_H
