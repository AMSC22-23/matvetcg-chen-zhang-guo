
/*
 * test_mbase.cpp
 *
 *  Created on: Jan 1, 2024
 *      Author: Ying Zhang
 */

#include <cstring>
#include <filesystem>
#include <cstddef>
#include <iostream>
#include <chrono>
#include "spai_mbase.hpp"

#include "Matrix/Matrix.hpp"
#include "MatrixWithVecSupport.hpp"
#include "Vector.hpp"
#include "cg.hpp"
#include "bicgstab.hpp"
#include "tools.hpp"

using std::cout;
using std::endl;


int main(int argc, char *argv[]) {
    using namespace apsc::LinearAlgebra;

    // Check if parameters are provided in the command line arguments
    if (argc != 4) {
        std::cout << "To run the program :\n";
        std::cerr << "Usage: " << argv[0] << " <filename> <max_iter> <epsilon>" << std::endl  // << "<n> is the the number of row/column of matrix A" << std::endl
                  << "<max_iter> is the maximal number of iterations to limit fill-in per column in M" << std::endl
                  << "<epsilon> is the stopping criterion on ||r||2 for every column of M" << std::endl;
        return 1; // Exit with an error code
    }

    std::string path = std::filesystem::current_path().string() + "/inputs/";
    path = path + argv[1];

    // Check if the file is opened successfully
    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl 
                  << "The mtx file should be placed in the ** src/inputs ** directory " << std::endl;
        return 1; // Exit with an error code
    }
    inputFile.close();


    std::cout << "Reading matrix A......" << std::endl;
    MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> A;
    Tools::read_mtx_matrix<decltype(A)>(A, path);
    const unsigned size = A.rows();
    std::cout << "A has been loaded successfully" << std::endl;
    // std::cout << "matrix A:\n" << A;

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
    MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> AT(size,size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            AT(i,j) = A(j,i);
        }
    }
    // std::cout << "Norm of skew-symmetric part: " << (AT-A).norm() << std::endl;
    
    // get M
    std::cout << "\nCreating the Matrix M(THE PRECONDITIONING OF A WITH SPARSE APPROXIMATE INVERSES)......" << std::endl;
    MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> M(size, size);
    int max_iter; 
    std::stringstream(argv[2]) >> max_iter;
    double epsilon;   
    std::stringstream(argv[3]) >> epsilon;    
    auto start_time = std::chrono::high_resolution_clock::now();
    LinearAlgebra::SPAI_MBASE<decltype(A), decltype(epsilon)>(A, M, max_iter, epsilon);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken by SPAI_MBASE: " << duration.count() << " microseconds" << std::endl;
    // std::cout << "matrix M:\n" << M << "\n";
    // std::cout << "\nM * A:\n" << Tools::multiply_two_matrix(M, A) << std::endl;
    // std::cout << "(M*A-identityMatrix).norm() =  "<< (M*A-identityMatrix).norm() << std::endl;

    // identityMatrix
    MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR> identityMatrix(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            identityMatrix(i,j) = 0;
            if (i==j) { identityMatrix(i,j)=1; }
        }
    }

    std::cout << "\nUsing iterative solvers......" << std::endl;


    // 1) with hand-made CG and identityMatrix
    MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR> AA(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            AA(i,j) = A(i,j);
        }
    }
    const Vector e(size, static_cast<double>(1));
    const Vector b = AA * e;
    Vector x(size, static_cast<double>(0));
    double tol = 1.e-8;
    int maxit = 1000;
    int result = LinearAlgebra::CG<decltype(AA), decltype(x), decltype(identityMatrix), decltype(tol)>(AA, x, b, identityMatrix, maxit, tol);        // Solve system
    std::cout << "--hand-made CG with identityMatrix: "<< std::endl;
    std::cout << "CG flag = " << result << std::endl;
    std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
    std::cout << "effective error =  "<<(x-e).norm()<< std::endl;


    // 2) with hand-made BiCGSTAB and identityMatrix
    x = x * 0.0;
    maxit = 1000;
    result = LinearAlgebra::BiCGSTAB<decltype(AA), decltype(x), decltype(identityMatrix), decltype(tol)>(AA, x, b, identityMatrix, maxit, tol);        // Solve system
    std::cout << "--hand-made BiCGSTAB with identityMatrix: "<< std::endl;
    std::cout << "BiCGSTAB flag = " << result << std::endl;
    std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
    std::cout << "effective error =  "<<(x-e).norm()<< std::endl;


    // 3) with hand-made CG and M
    MatrixWithVecSupport<double, Vector<double>, ORDERING::ROWMAJOR> MM(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            MM(i,j) = M(i,j);
        }
    }
    x = x * 0.0;
    maxit = 1000;
    AA = Tools::multiply_two_matrix(MM, AA);
    const Vector bb = MM * b;
    result = LinearAlgebra::CG<decltype(AA), decltype(x), decltype(MM), decltype(tol)>(AA, x, bb, MM, maxit, tol);        // Solve system
    std::cout << "--hand-made CG with SPAI: "<< std::endl;
    std::cout << "CG flag = " << result << std::endl;
    std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
    std::cout << "effective error =  "<<(x-e).norm()<< std::endl;


    // 4) with hand-made BiCGSTAB and M
    x = x * 0.0;
    maxit = 1000;
    result = LinearAlgebra::BiCGSTAB<decltype(AA), decltype(x), decltype(MM), decltype(tol)>(AA, x, bb, MM, maxit, tol);        // Solve system
    std::cout << "--hand-made BiCGSTAB with SPAI: "<< std::endl;
    std::cout << "BiCGSTAB flag = " << result << std::endl;
    std::cout << "maxit = 1000, iterations performed = " << maxit << std::endl;
    std::cout << "effective error =  "<<(x-e).norm()<< std::endl;


    return 0;
}
