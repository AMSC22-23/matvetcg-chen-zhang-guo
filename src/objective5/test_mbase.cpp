/*
 * test_mbase.hpp
 *
 *  Created on: Jan 1, 2024
 *      Author: Ying Zhang
 */

#include <cstddef>
#include <iostream>
#include <chrono>
#include "spai_mbase.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cstring>
#include <unsupported/Eigen/SparseExtra>

#include "Matrix/Matrix.hpp"
#include "MatrixWithVecSupport.hpp"

using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

using std::cout;
using std::endl;


int main(int argc, char *argv[]) {
    using namespace apsc::LinearAlgebra;

    std::cout << "Creating a random matrix A..." << std::endl;
    int n = 6;
    Matrix<double,ORDERING::ROWMAJOR> A(n,n);
    A.fillRandom();
    std::cout << "\nmatrix A:\n" << A;
    const unsigned size = A.rows();
    std::cout << "A has been created successfully" << std::endl;

    // Check A properties
    std::cout << "Matrix size:"<< A.rows() << " X " << A.cols() << std::endl;
    int nonZeros = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (A(i,j)!=0) { nonZeros+=1; }
        }
    }
    std::cout << "Non zero entries:" << nonZeros << std::endl;
    // Check symmetry
    Matrix<double,ORDERING::ROWMAJOR> AT(n,n);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            AT(i,j) = A(j,i);
        }
    }
    // std::cout << "Norm of skew-symmetric part: " << (AT-A).norm() << std::endl;

    // get M
    std::cout << "Creating the Matrix M(THE PRECONDITIONING OF A WITH SPARSE APPROXIMATE INVERSES)..." << std::endl;
    Matrix<double,ORDERING::ROWMAJOR> M(size, size);
    int max_iter = 10; 
    double epsilon = 0.6;    
    auto start_time = std::chrono::high_resolution_clock::now();
    LinearAlgebra::SPAI_MBASE<decltype(A), decltype(epsilon)>(A, M, max_iter, epsilon);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken by SPAI_MBASE: " << duration.count() << " microseconds" << std::endl;
    std::cout << "\nmatrix M:\n" << M << "\n\n";
    // std::cout << "\nM * A:\n" << M*A;
    Matrix<double,ORDERING::ROWMAJOR> identityMatrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identityMatrix(i,j) = 0;
            if (i==j) { identityMatrix(i,j)=1; }
        }
    }
    // std::cout << "(M*A-identityMatrix).norm() =  "<< (M*A-identityMatrix).norm() << std::endl;

    // 用Eigen的官方方法，类型不一致，用不了
    // 试试手搓的CG和CG_MPI

    // SpVec e = SpVec::Ones(size);
    // SpVec b = A * e;
    // SpVec x(size);
    // x = 0*x;
    // double tol = 1.e-4;
    // int result, maxit = 1000;
    // result = LinearAlgebra::CG_MODIFIED<decltype(A), decltype(x), decltype(tol)>(A, x, b, M, maxit, tol);        // Solve system
    // std::cout << "hand-made CG with SPAI "<< std::endl;
    // std::cout << "CG flag = " << result << std::endl;
    // std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
    // std::cout << "effective error =  "<<(x-e).norm()<< std::endl;

    


    return 0;
}
