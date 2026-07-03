#ifndef BASEMENTS_MATRIX3_H
#define BASEMENTS_MATRIX3_H

#include <cmath>
#include <algorithm>
#include "vec3.h"

// Forward declaration
namespace basements { namespace math { struct Quaternion; } }

namespace basements {
namespace math {

/**
 * @brief 3x3 matrix for rotations and linear transformations
 * 
 * Memory layout: Row-major (standard C/C++ array layout)
 * [m00 m01 m02]
 * [m10 m11 m12]
 * [m20 m21 m22]
 * 
 * Used for:
 * - Rotation matrices (SO(3))
 * - Inertia tensors
 * - Linear transformations
 */
struct alignas(16) Matrix3 {
    // Row-major storage
    float m[3][3];
    
    // ============================================================
    // Constructors
    // ============================================================
    
    /// Default constructor: identity matrix
    HOST_DEVICE Matrix3() {
        m[0][0] = 1.0f; m[0][1] = 0.0f; m[0][2] = 0.0f;
        m[1][0] = 0.0f; m[1][1] = 1.0f; m[1][2] = 0.0f;
        m[2][0] = 0.0f; m[2][1] = 0.0f; m[2][2] = 1.0f;
    }
    
    /// Construct from 9 elements (row-major)
    HOST_DEVICE Matrix3(float m00, float m01, float m02,
            float m10, float m11, float m12,
            float m20, float m21, float m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }
    
    /// Construct from column vectors
    HOST_DEVICE static Matrix3 from_columns(const Vec3& c0, const Vec3& c1, const Vec3& c2) {
        return Matrix3(
            c0.x, c1.x, c2.x,
            c0.y, c1.y, c2.y,
            c0.z, c1.z, c2.z
        );
    }
    
    /// Construct from row vectors
    HOST_DEVICE static Matrix3 from_rows(const Vec3& r0, const Vec3& r1, const Vec3& r2) {
        return Matrix3(
            r0.x, r0.y, r0.z,
            r1.x, r1.y, r1.z,
            r2.x, r2.y, r2.z
        );
    }
    
    /// Construct from quaternion
    HOST_DEVICE static Matrix3 from_quaternion(const Quaternion& q);
    
    /// Construct rotation matrix from axis-angle
    HOST_DEVICE static Matrix3 from_axis_angle(const Vec3& axis, float angle) {
        #ifdef __CUDA_ARCH__
        float c = cosf(angle);
        float s = sinf(angle);
        #else
        float c = std::cos(angle);
        float s = std::sin(angle);
        #endif
        float t = 1.0f - c;
        
        Vec3 a = axis.normalized();
        float x = a.x, y = a.y, z = a.z;
        
        return Matrix3(
            t*x*x + c,   t*x*y - s*z, t*x*z + s*y,
            t*x*y + s*z, t*y*y + c,   t*y*z - s*x,
            t*x*z - s*y, t*y*z + s*x, t*z*z + c
        );
    }
    
    /// Construct rotation matrix around X axis
    static Matrix3 rotation_x(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Matrix3(
            1.0f, 0.0f, 0.0f,
            0.0f, c,    -s,
            0.0f, s,     c
        );
    }
    
    /// Construct rotation matrix around Y axis
    static Matrix3 rotation_y(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Matrix3(
            c,    0.0f, s,
            0.0f, 1.0f, 0.0f,
            -s,   0.0f, c
        );
    }
    
    /// Construct rotation matrix around Z axis
    static Matrix3 rotation_z(float angle) {
        float c = std::cos(angle);
        float s = std::sin(angle);
        return Matrix3(
            c,    -s,   0.0f,
            s,     c,   0.0f,
            0.0f, 0.0f, 1.0f
        );
    }
    
    /// Construct scale matrix
    static Matrix3 scale(float sx, float sy, float sz) {
        return Matrix3(
            sx,   0.0f, 0.0f,
            0.0f, sy,   0.0f,
            0.0f, 0.0f, sz
        );
    }
    
    static Matrix3 scale(const Vec3& s) {
        return scale(s.x, s.y, s.z);
    }
    
    // ============================================================
    // Element Access
    // ============================================================
    
    HOST_DEVICE float* operator[](int row) { return m[row]; }
    HOST_DEVICE const float* operator[](int row) const { return m[row]; }
    
    HOST_DEVICE float& operator()(int row, int col) { return m[row][col]; }
    HOST_DEVICE const float& operator()(int row, int col) const { return m[row][col]; }
    
    // ============================================================
    // Column/Row Access
    // ============================================================
    
    Vec3 column(int col) const {
        return Vec3(m[0][col], m[1][col], m[2][col]);
    }
    
    Vec3 row(int row) const {
        return Vec3(m[row][0], m[row][1], m[row][2]);
    }
    
    void set_column(int col, const Vec3& v) {
        m[0][col] = v.x;
        m[1][col] = v.y;
        m[2][col] = v.z;
    }
    
    void set_row(int row, const Vec3& v) {
        m[row][0] = v.x;
        m[row][1] = v.y;
        m[row][2] = v.z;
    }
    
    // ============================================================
    // Arithmetic Operations
    // ============================================================
    
    HOST_DEVICE Matrix3 operator+(const Matrix3& other) const {
        Matrix3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.m[i][j] = m[i][j] + other.m[i][j];
            }
        }
        return result;
    }
    
    HOST_DEVICE Matrix3 operator-(const Matrix3& other) const {
        Matrix3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.m[i][j] = m[i][j] - other.m[i][j];
            }
        }
        return result;
    }
    
    HOST_DEVICE Matrix3 operator*(float scalar) const {
        Matrix3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.m[i][j] = m[i][j] * scalar;
            }
        }
        return result;
    }
    
    HOST_DEVICE Matrix3 operator/(float scalar) const {
        #ifdef __CUDA_ARCH__
        if (fabsf(scalar) < EPSILON) {
        #else
        if (std::abs(scalar) < EPSILON) {
        #endif
            return Matrix3();
        }
        return *this * (1.0f / scalar);
    }
    
    /// Matrix multiplication
    HOST_DEVICE Matrix3 operator*(const Matrix3& other) const {
        Matrix3 result;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                result.m[i][j] = 
                    m[i][0] * other.m[0][j] +
                    m[i][1] * other.m[1][j] +
                    m[i][2] * other.m[2][j];
            }
        }
        return result;
    }
    
    /// Matrix-vector multiplication
    HOST_DEVICE Vec3 operator*(const Vec3& v) const {
        return Vec3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
    
    // ============================================================
    // Matrix Operations
    // ============================================================
    
    /// Transpose
    HOST_DEVICE Matrix3 transposed() const {
        return Matrix3(
            m[0][0], m[1][0], m[2][0],
            m[0][1], m[1][1], m[2][1],
            m[0][2], m[1][2], m[2][2]
        );
    }
    
    HOST_DEVICE void transpose() {
        *this = transposed();
    }
    
    /// Determinant
    HOST_DEVICE float determinant() const {
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
             - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
             + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    }
    
    /// Inverse (using cofactor method)
    HOST_DEVICE Matrix3 inversed() const {
        float det = determinant();
        #ifdef __CUDA_ARCH__
        if (fabsf(det) < EPSILON) {
        #else
        if (std::abs(det) < EPSILON) {
        #endif
            return Matrix3();  // Return identity for singular matrix
        }
        
        float inv_det = 1.0f / det;
        
        Matrix3 result;
        result.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
        result.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
        result.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
        
        result.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
        result.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
        result.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
        
        result.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
        result.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
        result.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
        
        return result;
    }
    
    HOST_DEVICE void inverse() {
        *this = inversed();
    }
    
    /// Trace (sum of diagonal elements)
    HOST_DEVICE float trace() const {
        return m[0][0] + m[1][1] + m[2][2];
    }
    
    // ============================================================
    // Rotation Matrix Specific
    // ============================================================
    
    /// Check if matrix is orthogonal (rotation matrix)
    HOST_DEVICE bool is_orthogonal(float epsilon = EPSILON) const {
        Matrix3 mt = transposed();
        Matrix3 product = (*this) * mt;
        
        // Should be identity
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                float expected = (i == j) ? 1.0f : 0.0f;
                #ifdef __CUDA_ARCH__
                if (fabsf(product.m[i][j] - expected) > epsilon) {
                #else
                if (std::abs(product.m[i][j] - expected) > epsilon) {
                #endif
                    return false;
                }
            }
        }
        return true;
    }
    
    /// Convert to quaternion (assumes rotation matrix)
    HOST_DEVICE Quaternion to_quaternion() const;
    
    // ============================================================
    // Inertia Tensor Operations
    // ============================================================
    
    /// Construct inertia tensor for box
    static Matrix3 inertia_box(float mass, float width, float height, float depth) {
        float w2 = width * width;
        float h2 = height * height;
        float d2 = depth * depth;
        float m12 = mass / 12.0f;
        
        return Matrix3(
            m12 * (h2 + d2), 0.0f,             0.0f,
            0.0f,            m12 * (w2 + d2),  0.0f,
            0.0f,            0.0f,             m12 * (w2 + h2)
        );
    }
    
    /// Construct inertia tensor for sphere
    static Matrix3 inertia_sphere(float mass, float radius) {
        float i = 0.4f * mass * radius * radius;  // 2/5 * m * r²
        return Matrix3(
            i,    0.0f, 0.0f,
            0.0f, i,    0.0f,
            0.0f, 0.0f, i
        );
    }
    
    /// Construct inertia tensor for cylinder (along Z axis)
    static Matrix3 inertia_cylinder(float mass, float radius, float height) {
        float r2 = radius * radius;
        float h2 = height * height;
        float ixy = mass * (3.0f * r2 + h2) / 12.0f;
        float iz = 0.5f * mass * r2;
        
        return Matrix3(
            ixy,  0.0f, 0.0f,
            0.0f, ixy,  0.0f,
            0.0f, 0.0f, iz
        );
    }
    
    // ============================================================
    // Utility
    // ============================================================
    
    bool approx_equal(const Matrix3& other, float epsilon = EPSILON) const {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (std::abs(m[i][j] - other.m[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        return true;
    }
    
    // ============================================================
    // Static Factory Methods
    // ============================================================
    
    HOST_DEVICE static Matrix3 identity() {
        return Matrix3();
    }
    
    HOST_DEVICE static Matrix3 zero() {
        return Matrix3(
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f
        );
    }

    /// Outer product of two vectors (a * b^T)
    HOST_DEVICE static Matrix3 outer_product(const Vec3& a, const Vec3& b) {
        return Matrix3(
            a.x * b.x, a.x * b.y, a.x * b.z,
            a.y * b.x, a.y * b.y, a.y * b.z,
            a.z * b.x, a.z * b.y, a.z * b.z
        );
    }
};

// ============================================================
// Non-member operators
// ============================================================

HOST_DEVICE inline Matrix3 operator*(float scalar, const Matrix3& mat) {
    return mat * scalar;
}

} // namespace math
} // namespace basements

// Include quaternion header for conversion functions
#include "quaternion.h"

namespace basements {
namespace math {

// Implementation of Matrix3-Quaternion conversions
HOST_DEVICE inline Matrix3 Matrix3::from_quaternion(const Quaternion& q) {
    float xx = q.x * q.x;
    float yy = q.y * q.y;
    float zz = q.z * q.z;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float yz = q.y * q.z;
    float wx = q.w * q.x;
    float wy = q.w * q.y;
    float wz = q.w * q.z;
    
    return Matrix3(
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),        2.0f * (xz + wy),
        2.0f * (xy + wz),        1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
        2.0f * (xz - wy),        2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy)
    );
}

HOST_DEVICE inline Quaternion Matrix3::to_quaternion() const {
    float trace = m[0][0] + m[1][1] + m[2][2];
    
    if (trace > 0.0f) {
        #ifdef __CUDA_ARCH__
        float s = sqrtf(trace + 1.0f) * 2.0f;
        #else
        float s = std::sqrt(trace + 1.0f) * 2.0f;
        #endif
        return Quaternion(
            0.25f * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s
        );
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        #ifdef __CUDA_ARCH__
        float s = sqrtf(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        #else
        float s = std::sqrt(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f;
        #endif
        return Quaternion(
            (m[2][1] - m[1][2]) / s,
            0.25f * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s
        );
    } else if (m[1][1] > m[2][2]) {
        #ifdef __CUDA_ARCH__
        float s = sqrtf(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        #else
        float s = std::sqrt(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f;
        #endif
        return Quaternion(
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25f * s,
            (m[1][2] + m[2][1]) / s
        );
    } else {
        #ifdef __CUDA_ARCH__
        float s = sqrtf(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        #else
        float s = std::sqrt(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f;
        #endif
        return Quaternion(
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25f * s
        );
    }
}

} // namespace math
} // namespace basements

#endif // BASEMENTS_MATRIX3_H
