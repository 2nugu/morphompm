#ifndef MORPHOMPM_QUATERNION_H
#define MORPHOMPM_QUATERNION_H

#include <cmath>
#include <algorithm>
#include "vec3.h"

namespace morphompm {
namespace math {

/**
 * @brief Quaternion for 3D rotations
 * 
 * Representation: q = w + xi + yj + zk
 * where i² = j² = k² = ijk = -1
 * 
 * Memory layout: (w, x, y, z) - scalar first for mathematical convention
 * Unit quaternion: |q| = 1 (represents rotation)
 */
struct alignas(16) Quaternion {
    float w, x, y, z;  // w is scalar part, (x,y,z) is vector part
    

    // ============================================================
    // Constructors
    // ============================================================
    
    /// Default constructor: identity rotation
    HOST_DEVICE Quaternion() : w(1.0f), x(0.0f), y(0.0f), z(0.0f) {}
    
    /// Parameterized constructor
    HOST_DEVICE Quaternion(float w_, float x_, float y_, float z_) 
        : w(w_), x(x_), y(y_), z(z_) {}
    
    /// Construct from axis-angle (axis must be normalized)
    HOST_DEVICE static Quaternion from_axis_angle(const Vec3& axis, float angle) {
        float half_angle = angle * 0.5f;
        #ifdef __CUDA_ARCH__
        float s = sinf(half_angle);
        float c = cosf(half_angle);
        #else
        float s = std::sin(half_angle);
        float c = std::cos(half_angle);
        #endif
        
        return Quaternion(c, axis.x * s, axis.y * s, axis.z * s);
    }
    
    /// Construct from Euler angles (in radians, ZYX order)
    HOST_DEVICE static Quaternion from_euler(float pitch, float yaw, float roll) {
        #ifdef __CUDA_ARCH__
        float cy = cosf(yaw * 0.5f);
        float sy = sinf(yaw * 0.5f);
        float cp = cosf(pitch * 0.5f);
        float sp = sinf(pitch * 0.5f);
        float cr = cosf(roll * 0.5f);
        float sr = sinf(roll * 0.5f);
        #else
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);
        #endif
        
        return Quaternion(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        );
    }
    
    /// Construct rotation from one vector to another
    HOST_DEVICE static Quaternion from_to_rotation(const Vec3& from, const Vec3& to) {
        Vec3 from_norm = from.normalized();
        Vec3 to_norm = to.normalized();
        
        float dot = from_norm.dot(to_norm);
        
        // Vectors are parallel
        if (dot > 0.9999f) {
            return Quaternion();  // Identity
        }
        
        // Vectors are opposite
        if (dot < -0.9999f) {
            // Find orthogonal axis
            Vec3 axis = Vec3::unit_x().cross(from_norm);
            if (axis.length_squared() < EPSILON) {
                axis = Vec3::unit_y().cross(from_norm);
            }
            return from_axis_angle(axis.normalized(), PI);
        }
        
        // General case
        Vec3 axis = from_norm.cross(to_norm);
        #ifdef __CUDA_ARCH__
        float s = sqrtf((1.0f + dot) * 2.0f);
        #else
        float s = std::sqrt((1.0f + dot) * 2.0f);
        #endif
        float inv_s = 1.0f / s;
        
        return Quaternion(
            s * 0.5f,
            axis.x * inv_s,
            axis.y * inv_s,
            axis.z * inv_s
        );
    }
    
    // ============================================================
    // Arithmetic Operations
    // ============================================================
    
    HOST_DEVICE Quaternion operator+(const Quaternion& other) const {
        return Quaternion(w + other.w, x + other.x, y + other.y, z + other.z);
    }
    
    HOST_DEVICE Quaternion& operator+=(const Quaternion& other) {
        w += other.w;
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
    
    HOST_DEVICE Quaternion operator-(const Quaternion& other) const {
        return Quaternion(w - other.w, x - other.x, y - other.y, z - other.z);
    }
    
    HOST_DEVICE Quaternion operator*(float scalar) const {
        return Quaternion(w * scalar, x * scalar, y * scalar, z * scalar);
    }
    
    HOST_DEVICE Quaternion operator/(float scalar) const {
        #ifdef __CUDA_ARCH__
        if (fabsf(scalar) < EPSILON) {
        #else
        if (std::abs(scalar) < EPSILON) {
        #endif
            return Quaternion();
        }
        float inv = 1.0f / scalar;
        return *this * inv;
    }
    
    /// Quaternion multiplication (Hamilton product)
    HOST_DEVICE Quaternion operator*(const Quaternion& other) const {
        return Quaternion(
            w * other.w - x * other.x - y * other.y - z * other.z,
            w * other.x + x * other.w + y * other.z - z * other.y,
            w * other.y - x * other.z + y * other.w + z * other.x,
            w * other.z + x * other.y - y * other.x + z * other.w
        );
    }
    
    HOST_DEVICE Quaternion operator-() const {
        return Quaternion(-w, -x, -y, -z);
    }
    
    // Compound assignment
    HOST_DEVICE Quaternion& operator*=(const Quaternion& other) {
        *this = *this * other;
        return *this;
    }
    
    HOST_DEVICE Quaternion& operator*=(float scalar) {
        *this = *this * scalar;
        return *this;
    }
    
    // ============================================================
    // Quaternion Operations
    // ============================================================
    
    /// Conjugate: q* = w - xi - yj - zk
    HOST_DEVICE Quaternion conjugate() const {
        return Quaternion(w, -x, -y, -z);
    }
    
    /// Norm squared: |q|²
    HOST_DEVICE float norm_squared() const {
        return w * w + x * x + y * y + z * z;
    }
    
    /// Norm: |q|
    HOST_DEVICE float norm() const {
        #ifdef __CUDA_ARCH__
        return sqrtf(norm_squared());
        #else
        return std::sqrt(norm_squared());
        #endif
    }
    
    /// Normalize to unit quaternion
    HOST_DEVICE Quaternion normalized() const {
        float n = norm();
        if (n < EPSILON) {
            return Quaternion();  // Return identity
        }
        return *this / n;
    }
    
    HOST_DEVICE void normalize() {
        *this = normalized();
    }
    
    /// Inverse: q⁻¹ = q* / |q|²
    HOST_DEVICE Quaternion inverse() const {
        float n_sq = norm_squared();
        if (n_sq < EPSILON) {
            return Quaternion();
        }
        return conjugate() / n_sq;
    }
    
    /// Dot product
    HOST_DEVICE float dot(const Quaternion& other) const {
        return w * other.w + x * other.x + y * other.y + z * other.z;
    }
    
    // ============================================================
    // Rotation Operations
    // ============================================================
    
    /// Rotate a vector by this quaternion
    HOST_DEVICE Vec3 rotate(const Vec3& v) const {
        // v' = q * v * q⁻¹
        // Optimized version using vector algebra
        Vec3 qvec(x, y, z);
        Vec3 uv = qvec.cross(v);
        Vec3 uuv = qvec.cross(uv);
        
        return v + (uv * w + uuv) * 2.0f;
    }
    
    /// Get rotation axis (for unit quaternion)
    HOST_DEVICE Vec3 axis() const {
        float s_squared = 1.0f - w * w;
        if (s_squared < EPSILON) {
            return Vec3::unit_x();  // Arbitrary axis for identity rotation
        }
        
        #ifdef __CUDA_ARCH__
        float s = sqrtf(s_squared);
        #else
        float s = std::sqrt(s_squared);
        #endif
        return Vec3(x / s, y / s, z / s);
    }
    
    /// Get rotation angle (in radians)
    HOST_DEVICE float angle() const {
        #ifdef __CUDA_ARCH__
        return 2.0f * acosf(clamp(w, -1.0f, 1.0f));
        #else
        return 2.0f * std::acos(std::clamp(w, -1.0f, 1.0f));
        #endif
    }
    
    /// Convert to Euler angles (in radians, ZYX order)
    HOST_DEVICE void to_euler(float& pitch, float& yaw, float& roll) const {
        // Roll (x-axis rotation)
        float sinr_cosp = 2.0f * (w * x + y * z);
        float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
        #ifdef __CUDA_ARCH__
        roll = atan2f(sinr_cosp, cosr_cosp);
        #else
        roll = std::atan2(sinr_cosp, cosr_cosp);
        #endif
        
        // Pitch (y-axis rotation)
        float sinp = 2.0f * (w * y - z * x);
        #ifdef __CUDA_ARCH__
        if (fabsf(sinp) >= 1.0f) {
            pitch = copysignf(PI / 2.0f, sinp);  // Use 90 degrees
        } else {
            pitch = asinf(sinp);
        }
        #else
        if (std::abs(sinp) >= 1.0f) {
            pitch = std::copysign(PI / 2.0f, sinp);  // Use 90 degrees
        } else {
            pitch = std::asin(sinp);
        }
        #endif
        
        // Yaw (z-axis rotation)
        float siny_cosp = 2.0f * (w * z + x * y);
        float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
        #ifdef __CUDA_ARCH__
        yaw = atan2f(siny_cosp, cosy_cosp);
        #else
        yaw = std::atan2(siny_cosp, cosy_cosp);
        #endif
    }
    
    // ============================================================
    // Interpolation
    // ============================================================
    
    /// Linear interpolation (not recommended for rotations)
    HOST_DEVICE static Quaternion lerp(const Quaternion& a, const Quaternion& b, float t) {
        return (a * (1.0f - t) + b * t).normalized();
    }
    
    /// Spherical linear interpolation (SLERP)
    HOST_DEVICE static Quaternion slerp(const Quaternion& a, const Quaternion& b, float t) {
        Quaternion qa = a.normalized();
        Quaternion qb = b.normalized();
        
        float dot = qa.dot(qb);
        
        // If dot < 0, negate one quaternion to take shorter path
        if (dot < 0.0f) {
            qb = -qb;
            dot = -dot;
        }
        
        // If quaternions are very close, use linear interpolation
        if (dot > 0.9995f) {
            return lerp(qa, qb, t);
        }
        
        // Compute angle and sin
        #ifdef __CUDA_ARCH__
        float theta = acosf(dot);
        float sin_theta = sinf(theta);
        float wa = sinf((1.0f - t) * theta) / sin_theta;
        float wb = sinf(t * theta) / sin_theta;
        #else
        float theta = std::acos(dot);
        float sin_theta = std::sin(theta);
        float wa = std::sin((1.0f - t) * theta) / sin_theta;
        float wb = std::sin(t * theta) / sin_theta;
        #endif
        
        return qa * wa + qb * wb;
    }
    
    // ============================================================
    // Utility
    // ============================================================
    
    HOST_DEVICE bool is_identity(float epsilon = EPSILON) const {
        #ifdef __CUDA_ARCH__
        return fabsf(w - 1.0f) < epsilon &&
               fabsf(x) < epsilon &&
               fabsf(y) < epsilon &&
               fabsf(z) < epsilon;
        #else
        return std::abs(w - 1.0f) < epsilon &&
               std::abs(x) < epsilon &&
               std::abs(y) < epsilon &&
               std::abs(z) < epsilon;
        #endif
    }
    
    HOST_DEVICE bool approx_equal(const Quaternion& other, float epsilon = EPSILON) const {
        #ifdef __CUDA_ARCH__
        return fabsf(w - other.w) < epsilon &&
               fabsf(x - other.x) < epsilon &&
               fabsf(y - other.y) < epsilon &&
               fabsf(z - other.z) < epsilon;
        #else
        return std::abs(w - other.w) < epsilon &&
               std::abs(x - other.x) < epsilon &&
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon;
        #endif
    }
    
    // ============================================================
    // Static Factory Methods
    // ============================================================
    
    HOST_DEVICE static Quaternion identity() { return Quaternion(1.0f, 0.0f, 0.0f, 0.0f); }
    
    /// Rotation around X axis
    HOST_DEVICE static Quaternion rotation_x(float angle) {
        return from_axis_angle(Vec3::unit_x(), angle);
    }
    
    /// Rotation around Y axis
    HOST_DEVICE static Quaternion rotation_y(float angle) {
        return from_axis_angle(Vec3::unit_y(), angle);
    }
    
    /// Rotation around Z axis
    HOST_DEVICE static Quaternion rotation_z(float angle) {
        return from_axis_angle(Vec3::unit_z(), angle);
    }
    
    /// Look rotation (forward direction and up vector)
    HOST_DEVICE static Quaternion look_rotation(const Vec3& forward, const Vec3& up = Vec3::unit_y()) {
        Vec3 f = forward.normalized();
        Vec3 r = up.cross(f).normalized();
        Vec3 u = f.cross(r);
        
        // Convert rotation matrix to quaternion
        float trace = r.x + u.y + f.z;
        
        if (trace > 0.0f) {
            #ifdef __CUDA_ARCH__
            float s = sqrtf(trace + 1.0f) * 2.0f;
            #else
            float s = std::sqrt(trace + 1.0f) * 2.0f;
            #endif
            return Quaternion(
                0.25f * s,
                (u.z - f.y) / s,
                (f.x - r.z) / s,
                (r.y - u.x) / s
            );
        } else if (r.x > u.y && r.x > f.z) {
            #ifdef __CUDA_ARCH__
            float s = sqrtf(1.0f + r.x - u.y - f.z) * 2.0f;
            #else
            float s = std::sqrt(1.0f + r.x - u.y - f.z) * 2.0f;
            #endif
            return Quaternion(
                (u.z - f.y) / s,
                0.25f * s,
                (u.x + r.y) / s,
                (f.x + r.z) / s
            );
        } else if (u.y > f.z) {
            #ifdef __CUDA_ARCH__
            float s = sqrtf(1.0f + u.y - r.x - f.z) * 2.0f;
            #else
            float s = std::sqrt(1.0f + u.y - r.x - f.z) * 2.0f;
            #endif
            return Quaternion(
                (f.x - r.z) / s,
                (u.x + r.y) / s,
                0.25f * s,
                (f.y + u.z) / s
            );
        } else {
            #ifdef __CUDA_ARCH__
            float s = sqrtf(1.0f + f.z - r.x - u.y) * 2.0f;
            #else
            float s = std::sqrt(1.0f + f.z - r.x - u.y) * 2.0f;
            #endif
            return Quaternion(
                (r.y - u.x) / s,
                (f.x + r.z) / s,
                (f.y + u.z) / s,
                0.25f * s
            );
        }
    }
};

// ============================================================
// Non-member operators
// ============================================================

HOST_DEVICE inline Quaternion operator*(float scalar, const Quaternion& q) {
    return q * scalar;
}

HOST_DEVICE inline bool operator==(const Quaternion& lhs, const Quaternion& rhs) {
    return lhs.w == rhs.w && lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

HOST_DEVICE inline bool operator!=(const Quaternion& lhs, const Quaternion& rhs) {
    return !(lhs == rhs);
}

} // namespace math
} // namespace morphompm

#endif // MORPHOMPM_QUATERNION_H
