/*
 * spai_openmp.hpp
 *
 *  Created on: Dec 17, 2023
 *      Author: Ying Zhang
 */

#ifndef HH_SPAI_OPENMP__HH
#define HH_SPAI_OPENMP__HH
//*****************************************************************
// SParse Approximate Inverse algorithm - SPAI supporting Eigen::SparseMatrix and OpenMP
// 
// Every thread in OpenMP executes one to several columns of matrix M in parallel.
//
// The SPAI algorithm computes a sparse approximate inverse M of a general 
// sparse matrix A. It is inherently parallel, since the columns of M are 
// calculated independently of one another.
//
// Grote M J, Huckle T. Parallel preconditioning with sparse 
// approximate inverses[J]. SIAM Journal on Scientific Computing, 
// 1997, 18(3): 838-853.
//
// Abstract. 
// A parallel preconditioner is presented for the solution of 
// general sparse linear systems of equations. A sparse approximate 
// inverse is computed explicitly and then applied as a preconditioner 
// to an iterative method. The computation of the preconditioner is 
// inherently parallel, and its application only requires a 
// matrix-vector product. The sparsity pattern of the approximate 
// inverse is not imposed a priori but captured automatically. This keeps 
// the amount of work and the number of nonzero entries in the 
// preconditioner to a minimum. Rigorous bounds on the clustering of 
// the eigenvalues and the singular values are derived for the 
// preconditioned system, and the proximity of the approximate to the 
// true inverse is estimated.
// 
// Upon successful return, output arguments have the following values:
//
//        A  --  matrix A in equation Ax=b
//        M  --  the sparse approximate inverse of matrix A
// max_iter  --  the number of iterations performed to limit 
//               the maximal fill-in per column in M
//  epsilon  --  the threshold of stopping criterion on residual ||r||
//               for every column of M
//
//*****************************************************************

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include <cstddef>
#include "Matrix/Matrix.hpp"
#include "MatrixWithVecSupport.hpp"
#include <omp.h>

#include <memory>

/*! 
 * Extract a block from matrix `A` specified by the indices set provided in 
 * vectors `I` and `J` and stores the result in the Eigen::MatrixXd `AIJ`.
 */
template <class Matrix>
void getBlockByTwoIndicesSet(const Matrix&A, Eigen::MatrixXd &AIJ, 
    const std::vector<int> &I, const std::vector<int> &J) {

    int n1 = I.size();
    int n2 = J.size();
    for (int i=0; i<n1; i++) {
        for (int j=0; j<n2; j++) {
            AIJ(i,j) = A.coeff(I[i], J[j]);
        }
    }

}

/*!
 * Performs the QR decomposition of a given matrix `A` and returns 
 * the results in matrices `Q` and `R`. The QR decomposition is achieved 
 * using the ColPivHouseholderQR decomposition method provided 
 * by the Eigen library.
 */
void qrDecomposition(const Eigen::MatrixXd &A, Eigen::MatrixXd &Q, Eigen::MatrixXd &R) {

    // When choosing a QR decomposition method, you can usually consider the following factors:
    // Performance: The performance of different methods may vary depending on the nature of the matrix. ColPivHouseholderQR generally performs well in most situations.
    // Memory usage: FullPivHouseholderQR provides full QR decomposition, but may require more memory.
    // Numerical stability: HouseholderQR may be more stable in some situations where numerical values are worse.
    // Function: Consider your specific task needs and choose the appropriate QR decomposition method.
    std::unique_ptr<Eigen::HouseholderQR<Eigen::MatrixXd> > qr_p = std::make_unique<Eigen::HouseholderQR<Eigen::MatrixXd> >(A);
    Q = qr_p->householderQ();
    R = qr_p->matrixQR().topLeftCorner(A.cols(), A.cols());

}

/*!
 * According to the process required in the paper, compute k-th column `m_k` of M
 * and the corresponding residual `r`. This function will be used many times in the
 * entire algorithm, but the first computing for `r` is different from others for
 * every column of M so we use the variable `flag` to distinguish
 */
template <class Matrix, typename Scalar>
void computeRAndMk(const Matrix&A, const int Size, const Eigen::MatrixXd &Q, 
    const Eigen::MatrixXd &R, Eigen::VectorXd &r, Eigen::VectorXd &m_k,
    const Eigen::VectorXd &e_k, const std::vector<int> &I, const std::vector<int> &J, int flag) {
    
    int n1 = I.size();
    int n2 = J.size();
    // e_k_triangular = e_k(I)
    std::unique_ptr<Eigen::VectorXd> e_k_triangular_ptr = std::make_unique<Eigen::VectorXd>(n1);
    for (int i=0; i<n1; i++) {
       (*e_k_triangular_ptr)(i) = e_k[I[i]];
    }
    std::unique_ptr<Eigen::VectorXd> c_triangular_ptr = std::make_unique<Eigen::VectorXd>(n1);
    *c_triangular_ptr = Q.transpose() * (*e_k_triangular_ptr);
    std::unique_ptr<Eigen::VectorXd> m_k_triangular_ptr = std::make_unique<Eigen::VectorXd>(n2);
    *m_k_triangular_ptr = R.inverse() * c_triangular_ptr->segment(0,n2-1);
    // m_k
    int index = 0;
    for (int i=0; i<Size; i++) {
        if (index<n2 && J[index]==i) {
            m_k[i] = (*m_k_triangular_ptr)(index);
            index++;
        } else {
            m_k[i] = 0;
        }
    }
    // AJ be the A(.,J)
    std::unique_ptr<Eigen::MatrixXd> AJ_ptr = std::make_unique<Eigen::MatrixXd>(Size, n2);
    for (int i=0; i<Size; i++) {
        for (int j=0; j<n2; j++) {
            (*AJ_ptr)(i,j) = A.coeff(i, J[j]);
        }
    }
    // r
    if (flag==0) { r = (*AJ_ptr) * (*m_k_triangular_ptr) - e_k;} 
    if (flag==1) {
        // A * m_k - e_k
        std::unique_ptr<Eigen::VectorXd> m_k_vector_ptr = std::make_unique<Eigen::VectorXd>(Size);
        for (int i=0; i<Size; i++) { 
            (*m_k_vector_ptr)(i) = m_k[i]; 
        }
        std::unique_ptr<Eigen::VectorXd> intermediate_ptr = std::make_unique<Eigen::VectorXd>(Size);
        *intermediate_ptr = A * (*m_k_vector_ptr);
        r = *intermediate_ptr - e_k;
    } 

}


/*!
 * @note Scalar is usually double
 * @param max_iter is usually Size or you can set your own
 * @param epsilon can be selected from set {0.2, 0.3, 0.4, 0.5, 0.6} 
 */ 
namespace LinearAlgebra {
template <class Matrix, typename Scalar>
int SPAI_OPENMP(const Matrix &A, Matrix &M, const int max_iter, Scalar epsilon) {

        ASSERT((A.rows() == A.cols()), "The matrix must be squared!\n");
        const int Size = A.rows();

        // // strategy 1: the initial sparsity of M is chosen to be diagonal
        std::cout << "The initial sparsity of M which is chosen to be diagonal..." << std::endl;
        const Scalar diagonal_value = 1;
        const Scalar zero = 0;
        for (int i=0; i<Size; i++) {
            for (int j=0; j<Size; j++) {
                if (i==j) { M.insert(i, j) = diagonal_value; }
                else { M.insert(i,j) = zero; } 
            }
        }
        M.makeCompressed();
        // std::cout << "matrix M:\n" << M << std::endl;

        // // strategy 2: M is chosen to be not diagonal
        // std::cout << "The initial sparsity of M which is chosen to be lkie..." << std::endl;
        // const Scalar diagonal_value = 1;
        // const Scalar zero = 0;
        // for (int i=0; i<Size; i++) {
        //     for (int j=0; j<Size; j++) {
        //         if (i==j) { M.insert(i, j) = diagonal_value; }
        //         else if (i==j+1) { M.insert(i, j) = diagonal_value; }
        //         else { M.insert(i,j) = zero; } 
        //     }
        // }
        // M.makeCompressed();
        // // std::cout << "matrix M:\n" << M << std::endl;

        // parallel
        #pragma omp parallel for

        // for every column of M
        for (int k=0; k<Size; k++) {
            // (a) 
            // J be the set of indices j such that m_k(j) != 0
            std::vector<int> J;
            for (int j=0; j<Size; j++) {
                if (M.coeff(j,k)!=zero) { 
                    J.push_back(j); 
                }
            }
            // (b)
            // I be the set of indices i such that A(i, J) is not identically zero.
            std::vector<int> I;
            for (int i=0; i<Size; i++) {
                int flag = 0;
                for (const auto &j : J) { 
                    if (A.coeff(i,j)!=zero) {flag = 1;}
                }
                if (flag==1) { 
                    I.push_back(i);
                }
            }
            // AIJ be the A(I,J)
            int n1 = I.size();
            int n2 = J.size();
            ASSERT((n1 >= n2), "\nn1 must be bigger than or equal to n2!\n");
            std::unique_ptr<Eigen::MatrixXd> AIJ_ptr = std::make_unique<Eigen::MatrixXd>(n1, n2);
            getBlockByTwoIndicesSet(A, *AIJ_ptr, I, J);

            // QR decomposition of AIJ
            std::unique_ptr<Eigen::MatrixXd> Q_ptr = std::make_unique<Eigen::MatrixXd>(AIJ_ptr->rows(), AIJ_ptr->rows());
            std::unique_ptr<Eigen::MatrixXd> R_ptr = std::make_unique<Eigen::MatrixXd>(AIJ_ptr->cols(), AIJ_ptr->cols());
            qrDecomposition(*AIJ_ptr, *Q_ptr, *R_ptr);

            // e_k
            std::unique_ptr<Eigen::VectorXd> e_k_ptr = std::make_unique<Eigen::VectorXd>(Size);
            e_k_ptr->setZero();
            (*e_k_ptr)(k) = 1.0;

            // m_k
            std::unique_ptr<Eigen::VectorXd> m_k_ptr = std::make_unique<Eigen::VectorXd>(Size);
            m_k_ptr->setZero();
            // r
            std::unique_ptr<Eigen::VectorXd> r_ptr = std::make_unique<Eigen::VectorXd>(Size);
            r_ptr->setZero();
            computeRAndMk<decltype(A), decltype(epsilon)>(A, Size, *Q_ptr, *R_ptr, *r_ptr, *m_k_ptr, *e_k_ptr, I, J, 0);


            // while loop iteration 
            int iter = 0;
            while (r_ptr->norm() > epsilon && iter < max_iter) {
                iter++;

                // (c) 
                std::vector<int> L;
                // Strategy 1: Set L equal to the set of indices l for which r(l) != 0
                for (int i=0; i<Size; i++) {
                    if ((*r_ptr)(i)!=0) { 
                        L.push_back(i); 
                    }
                }
                // // Strategy 2: according to suggestion of remarks.5, choose the largest elements in r
                // int max_index = 0;
                // for (int i=0; i<Size; i++) {
                //     if (std::abs((*r_ptr)(i)) > std::abs((*r_ptr)(max_index))) { 
                //          max_index = i;
                //     }
                // }
                // L.push_back(max_index);

                // (d) J_triangular
                std::vector<int> J_triangular;
                int index = 0;
                for (int j=0; j<Size; j++) {
                    if (index<n2 && J[index] == j) {
                        index++;
                    } else {
                        int flag = 0;
                        for (const auto &i : L) {
                            if (A.coeff(i,j)!=zero) {
                                flag = 1;
                            }
                        }
                        if (flag==1) { 
                            J_triangular.push_back(j);
                        } 
                    }
                }

                // // (e) For each j ∈ J_triangular solve the minimization problem (10).
                // std::vector<Scalar> rou;
                // for (const auto &j : J_triangular) {
                //     Eigen::VectorXd e_j = Eigen::VectorXd::Zero(Size);
                //     e_j[j] = 1.0;
                //     Eigen::VectorXd aej = A * e_j;
                //     Eigen::VectorXd aej_ev = aej;
                //     Scalar m1 = r.dot(aej_ev);
                //     Scalar m2 = std::pow(aej_ev.norm(),2);
                //     Scalar miu_j = m1 / m2;
                //     Scalar rou_j = std::pow(r.norm(),2) + miu_j*(r.dot(aej_ev));
                //     rou.push_back(rou_j);
                // }
                // // (f)
                // // first: reserve indices j that rou_j is less than or equal to the mean value of all rou_j
                // Scalar sum_of_rou = 0;
                // for (const auto &r: rou) {
                //     sum_of_rou += r;
                // }
                // Scalar mean_of_rou = sum_of_rou / rou.size();
                // std::vector<int> J_triangular_first;
                // std::vector<Scalar> rou_first;
                // for (int j=0; j<J_triangular.size(); j++) {
                //     if (rou[j] <= mean_of_rou) { 
                //         J_triangular_first.push_back(J_triangular[j]); 
                //         rou_first.push_back(rou[j]);
                //     }
                // }
                // // second: From the remaining indices we keep at most s indices with smallest rou_j, and we set s equal to 5
                // std::vector<int> J_triangular_second;
                // std::vector<Scalar> rou_second;
                // if (rou_first.size() > 5) {
                //     // find the 5th smallest rou
                //     std::vector<Scalar> copy_rou_first = rou_first;
                //     std::partial_sort(copy_rou_first.begin(), copy_rou_first.begin() + 5, copy_rou_first.end());
                //     Scalar fifth_smallest_rou = copy_rou_first[4];
                //     // delete indices j that rou_j is bigger than or equal to the 5th smallest rou
                //     for (int j=0; j<J_triangular_first.size(); j++) {
                //         if (rou[j] <= fifth_smallest_rou) { 
                //             J_triangular_second.push_back(J_triangular_first[j]); 
                //             rou_second.push_back(rou_first[j]);
                //         }
                //     }
                //     J_triangular.swap(J_triangular_second);
                // } else {
                //     J_triangular.swap(J_triangular_first);
                // }
                

                // (g) Determine the new indices I_triangular
                std::vector<int> J_merged(J.size() + J_triangular.size());
                std::merge(J.begin(), J.end(), J_triangular.begin(), J_triangular.end(), J_merged.begin());
                std::sort(J_merged.begin(), J_merged.end());
                
                std::vector<int> I_triangular;
                index = 0;
                for (int i=0; i<Size; i++) {
                    if (index<n1 && I[index] == i) {
                        index++;
                    } else {
                        int flag = 0;
                        for (const auto &j : J_merged) { 
                            if (A.coeff(i,j)!=zero) {flag = 1;}
                        }
                        if (flag==1) { 
                            I_triangular.push_back(i);
                        } 
                    }
                }
                std::vector<int> I_merged(I.size() + I_triangular.size());
                std::merge(I.begin(), I.end(), I_triangular.begin(), I_triangular.end(), I_merged.begin());
                std::sort(I_merged.begin(), I_merged.end());

                int n1_triangular = I_triangular.size();
                int n2_triangular = J_triangular.size();


                // for updating Q and R at the end of this loop
                std::unique_ptr<Eigen::MatrixXd> Pr_ptr = std::make_unique<Eigen::MatrixXd>(n1+n1_triangular, n1+n1_triangular);
                Pr_ptr->setZero();
                std::vector<int> I_other;
                int ind = 0;
                for (int i = 0; i < Size; i++) {
                    if (ind < I_merged.size() && I_merged[ind] == i) {ind++;}
                    else {I_other.push_back(i);}
                }
                std::vector<int> I_copy(I);
                std::vector<int> I_triangular_copy(I_triangular);
                int flag;
                do {
                    flag = 0;
                    for (const auto &i : I_other) { 
                        for (auto &i1 : I_copy) {
                            if (i1 > i) {i1--;}
                            if (i1 < n1+n1_triangular) {flag+=0;} else {flag+=1;}
                        }
                        for (auto &i2 : I_triangular_copy) {
                            if (i2 > i) {i2--;}
                            if (i2 < n1+n1_triangular) {flag+=0;} else {flag+=1;}
                        }
                    }
                } while (flag>0);
                for (int i = 0; i < n1+n1_triangular; i++)
                {
                    if (i < n1) { (*Pr_ptr)(I_copy[i], i) = 1.0; }
                    else { (*Pr_ptr)(I_triangular_copy[i-n1], i) = 1.0; }
                }
            
                std::unique_ptr<Eigen::MatrixXd> Pc_ptr = std::make_unique<Eigen::MatrixXd>(n2+n2_triangular, n2+n2_triangular);
                Pc_ptr->setZero();
                std::vector<int> J_other;
                ind = 0;
                for (int i = 0; i < Size; i++) {
                    if (ind<J_merged.size() && J_merged[ind] == i) {ind++;}
                    else {J_other.push_back(i);}
                }
                std::vector<int> J_copy(J);
                std::vector<int> J_triangular_copy(J_triangular);
                do {
                    flag = 0;
                    for (const auto &j : J_other) { 
                        for (auto &j1 : J_copy) {
                            if (j1 > j) {j1--;}
                            if (j1 < n2+n2_triangular) {flag+=0;} else {flag+=1;}
                        }
                        for (auto &j2 : J_triangular_copy) {
                            if (j2 > j) {j2--;}
                            if (j2 < n2+n2_triangular) {flag+=0;} else {flag+=1;}
                        }
                    }
                } while (flag>0);
                for (int i = 0; i < n2+n2_triangular; i++)
                {
                    if (i < n2) { (*Pc_ptr)(i, J_copy[i]) = 1.0; }
                    else { (*Pc_ptr)(i, J_triangular_copy[i-n2]) = 1.0; }
                }
                
                //
                // A_I_J_triangular
                std::unique_ptr<Eigen::MatrixXd> A_I_J_triangular_ptr = std::make_unique<Eigen::MatrixXd>(n1, n2_triangular);
                getBlockByTwoIndicesSet(A, *A_I_J_triangular_ptr, I, J_triangular);
                // A_I_triangular_J_triangular
                std::unique_ptr<Eigen::MatrixXd> A_I_triangular_J_triangular_ptr = std::make_unique<Eigen::MatrixXd>(n1_triangular, n2_triangular);
                getBlockByTwoIndicesSet(A, *A_I_triangular_J_triangular_ptr, I_triangular, J_triangular);

                // B1 and B2
                std::unique_ptr<Eigen::MatrixXd> B1_ptr = std::make_unique<Eigen::MatrixXd>(n2, n2_triangular);
                *B1_ptr = Q_ptr->transpose().topLeftCorner(n2,n1) * (*A_I_J_triangular_ptr);
                std::unique_ptr<Eigen::MatrixXd> B2_ptr = std::make_unique<Eigen::MatrixXd>(n1_triangular+n1-n2, n2_triangular);
                if (n1_triangular==0) {
                    *B2_ptr << Q_ptr->transpose().bottomLeftCorner(n1-n2, n1) * (*A_I_J_triangular_ptr);
                } else {
                    *B2_ptr << Q_ptr->transpose().bottomLeftCorner(n1-n2, n1) * (*A_I_J_triangular_ptr), 
                    *A_I_triangular_J_triangular_ptr;
                }
                
                std::unique_ptr<Eigen::MatrixXd> Q_triangular_ptr = std::make_unique<Eigen::MatrixXd>(n1_triangular+n1-n2, n1_triangular+n1-n2);
                std::unique_ptr<Eigen::MatrixXd> R_triangular_ptr = std::make_unique<Eigen::MatrixXd>(n2_triangular, n2_triangular);
                qrDecomposition(*B2_ptr, *Q_triangular_ptr, *R_triangular_ptr);
                // Set elements below the diagonal to 0
                for (int i = 0; i < R_triangular_ptr->rows(); ++i) {
                    for (int j = 0; j < i; ++j) {
                        (*R_triangular_ptr)(i, j) = 0.0;
                    }
                }
                

                // to be used in the next part
                std::unique_ptr<Eigen::MatrixXd> zeroMatrix1_ptr = std::make_unique<Eigen::MatrixXd>(Size, Size);
                std::unique_ptr<Eigen::MatrixXd> zeroMatrix2_ptr = std::make_unique<Eigen::MatrixXd>(Size, Size);
                std::unique_ptr<Eigen::MatrixXd> identityMatrix_ptr = std::make_unique<Eigen::MatrixXd>(Size, Size);
                
                // Q_new and R_new is the QR decomposition of A(I+I_triangular, J+J_triangular)
                //
                std::unique_ptr<Eigen::MatrixXd> result01_ptr = std::make_unique<Eigen::MatrixXd>(n1+n1_triangular, n1+n1_triangular);
                zeroMatrix1_ptr->resize(n1, n1_triangular);
                zeroMatrix1_ptr->setZero();
                zeroMatrix2_ptr->resize(n1_triangular, n1);
                zeroMatrix2_ptr->setZero();
                identityMatrix_ptr->resize(n1_triangular, n1_triangular);
                identityMatrix_ptr->setIdentity();
                *result01_ptr << *Q_ptr, *zeroMatrix1_ptr,
                            *zeroMatrix2_ptr, *identityMatrix_ptr; 

                std::unique_ptr<Eigen::MatrixXd> result02_ptr = std::make_unique<Eigen::MatrixXd>(n1+n1_triangular, n1+n1_triangular);
                identityMatrix_ptr->resize(n2, n2);
                identityMatrix_ptr->setIdentity();
                zeroMatrix1_ptr->resize(n2, n1_triangular+n1-n2);
                zeroMatrix1_ptr->setZero();
                zeroMatrix2_ptr->resize(n1_triangular+n1-n2, n2);
                zeroMatrix2_ptr->setZero();
                *result02_ptr << *identityMatrix_ptr, *zeroMatrix1_ptr,
                            *zeroMatrix2_ptr, *Q_triangular_ptr;

                // 
                std::unique_ptr<Eigen::MatrixXd> Q_new_ptr = std::make_unique<Eigen::MatrixXd>(n1+n1_triangular, n1+n1_triangular);
                *Q_new_ptr = (*result01_ptr) * (*result02_ptr);
                // 
                std::unique_ptr<Eigen::MatrixXd> R_new_ptr = std::make_unique<Eigen::MatrixXd>(n2+n2_triangular, n2+n2_triangular);
                zeroMatrix1_ptr->resize(n2_triangular, n2);
                zeroMatrix1_ptr->setZero();
                *R_new_ptr << *R_ptr, *B1_ptr,
                        *zeroMatrix1_ptr, *R_triangular_ptr;

                // 
                Q_ptr=std::move(Q_new_ptr);
                R_ptr=std::move(R_new_ptr);
            
                // update I and J
                I.swap(I_merged);
                J.swap(J_merged);
                n1 = I.size();
                n2 = J.size();

                // repeate the squares problem process and update variables
                computeRAndMk<decltype(A), decltype(epsilon)>(A, Size, *Q_ptr, *R_ptr, *r_ptr, *m_k_ptr, *e_k_ptr, I, J, 1);

                // update Q and R to be used in the next iteration
                *Q_ptr = (*Pr_ptr) * (*Q_ptr);
                *R_ptr = (*R_ptr) * (*Pc_ptr);
                
                // one loop end
            }

            // M
            for(int i=0; i<Size; i++) {
                auto value = (*m_k_ptr)(i);
                if(std::isnan(value)) { M.coeffRef(i,k)=0; }
                else{ M.coeffRef(i,k)=value; }
            }

        }
        return 0;
    }
}

#endif