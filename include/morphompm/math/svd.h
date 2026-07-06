#pragma once

#include "morphompm/math/matrix3.h"
#include "morphompm/math/common.h"
#include <cmath>
#include <algorithm>

namespace morphompm {
namespace math {

    /**
     * @brief 3x3 Singular Value Decomposition (SVD)
     * F = U * Sigma * V^T
     */
    class SVD {
    public:
        // Compute SVD of F: F = U * Sigma * V^T
        HOST_DEVICE static void compute(const Matrix3& F, Matrix3& U, Vec3& Sigma, Matrix3& V) {
            // 1. Solve Eigenproblem for A = F^T * F
            Matrix3 A = F.transposed() * F;
            
            V = Matrix3::identity();
            
            const int MAX_SWEEPS = 5;
            for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
                float max_off_diag = 0.0f;
                
                // Iterate upper triangle
                for (int i = 0; i < 2; ++i) {
                    for (int j = i + 1; j < 3; ++j) {
                        float a_ij = A.m[i][j];
                        float a_ii = A.m[i][i];
                        float a_jj = A.m[j][j];
                        
                        max_off_diag = std::max(max_off_diag, std::abs(a_ij));
                        
                        if (std::abs(a_ij) < 1e-6f) continue;
                        
                        float tau = (a_jj - a_ii) / (2.0f * a_ij);
                        float t;
                        if (tau >= 0.0f) {
                            t = 1.0f / (tau + std::sqrt(1.0f + tau * tau));
                        } else {
                            t = -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
                        }
                        
                        float c = 1.0f / std::sqrt(1.0f + t * t);
                        float s = c * t;
                        
                        // Update A (Jacobi rotation) - approximate for speed or precise manually
                        // Manual update for numerical stability
                        float new_aii = c*c*a_ii - 2.0f*s*c*a_ij + s*s*a_jj;
                        float new_ajj = s*s*a_ii + 2.0f*s*c*a_ij + c*c*a_jj;
                        float new_aij = 0.0f;
                        
                        A.m[i][i] = new_aii;
                        A.m[j][j] = new_ajj;
                        A.m[i][j] = new_aij;
                        A.m[j][i] = new_aij;
                        
                        for (int k = 0; k < 3; ++k) {
                            if (k != i && k != j) {
                                float aik = A.m[i][k];
                                float ajk = A.m[j][k];
                                A.m[i][k] = c * aik - s * ajk;
                                A.m[k][i] = A.m[i][k];
                                A.m[j][k] = s * aik + c * ajk;
                                A.m[k][j] = A.m[j][k];
                            }
                        }
                        
                        // Update V
                        for (int k = 0; k < 3; ++k) {
                            float v_ki = V.m[k][i];
                            float v_kj = V.m[k][j];
                            
                            V.m[k][i] = c * v_ki - s * v_kj;
                            V.m[k][j] = s * v_ki + c * v_kj;
                        }
                    }
                }
                
                if (max_off_diag < 1e-5f) break;
            }
            
            float l0 = std::max(0.0f, A.m[0][0]);
            float l1 = std::max(0.0f, A.m[1][1]);
            float l2 = std::max(0.0f, A.m[2][2]);
            
            float s0 = std::sqrt(l0);
            float s1 = std::sqrt(l1);
            float s2 = std::sqrt(l2);
            
            Sigma = Vec3(s0, s1, s2);
            
            // 2. Compute U = F * V * Sigma_inv
            Matrix3 FV = F * V;
            
            Vec3 u0, u1, u2;
            
            if (s0 > 1e-6f) u0 = FV.column(0) / s0; else u0 = Vec3::unit_x(); 
            if (s1 > 1e-6f) u1 = FV.column(1) / s1; else u1 = Vec3::unit_y(); 
            if (s2 > 1e-6f) u2 = FV.column(2) / s2; else u2 = Vec3::unit_z();
            
            U.set_column(0, u0);
            U.set_column(1, u1);
            U.set_column(2, u2);
        }
    };

} // namespace math
} // namespace morphompm
