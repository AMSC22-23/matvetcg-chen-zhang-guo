
/*
 * test_openmp.cpp
 *
 *  Created on: Dec 26, 2023
 *      Author: Ying Zhang
 */

#include <cstring>
#include <filesystem>
#include <cstddef>
#include <iostream>
#include <chrono>
#include "spai_openmp.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

using std::cout;
using std::endl;


int main(int argc, char *argv[]) {
    using namespace apsc::LinearAlgebra;

    // Check if parameters are provided in the command line arguments
    if (argc != 4) {
        std::cout << "To run the program :\n";
        std::cerr << "Usage: " << argv[0] << " <filename> <max_iter> <epsilon>" << std::endl
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
    

    std::cout << "Reading matrix A..." << std::endl;
    SpMat A;
    Eigen::loadMarket(A, path);
    const unsigned size = A.rows();
    std::cout << "A has been loaded successfully" << std::endl;
    // std::cout << "\nmatrix A:\n" << A;

    // Check A properties
    std::cout << "Matrix size:"<< A.rows() << " X " << A.cols() << std::endl;
    std::cout << "Non zero entries:" << A.nonZeros() << std::endl;
    SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
    std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;

    // get M
    std::cout << "Creating the Matrix M(THE PRECONDITIONING OF A WITH SPARSE APPROXIMATE INVERSES)..." << std::endl;
    SpMat M(size, size);
    int max_iter; 
    std::stringstream(argv[2]) >> max_iter;
    double epsilon;   
    std::stringstream(argv[3]) >> epsilon;   
    auto start_time = std::chrono::high_resolution_clock::now();
    LinearAlgebra::SPAI_OPENMP<decltype(A), decltype(epsilon)>(A, M, max_iter, epsilon);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken by SPAI_OPENMP: " << duration.count() << " microseconds" << std::endl;
    // std::cout << "\nM * A:\n" << M*A;
    SpMat identityMatrix(size, size);
    identityMatrix.setIdentity();
    std::cout << "(M*A-identityMatrix).norm() =  "<< (M*A-identityMatrix ).norm() << std::endl;

    std::cout << "Using iterative solvers......" << std::endl;


    // 1) with Eigen CG
    const SpVec e = SpVec::Ones(size);
    SpVec b = A*e;
    SpVec x = SpVec::Zero(size);
    A = M * A;
    b = M * b;
    const double tol = 1.e-8;
    const int maxit = 1000;
    int result;
    
    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setMaxIterations(maxit);
    cg.setTolerance(tol);
    x = cg.compute(A).solve(b);

    std::cout << "--Eigen native CG" << std::endl;
    std::cout << "#iterations:       " << cg.iterations() << std::endl;
    std::cout << "relative residual: " << cg.error()      << std::endl;
    std::cout << "effective error:   " << (x-e).norm()    << std::endl;


    // 2) with Eigen BiCGSTAB
    x = 0 * x;
    
    Eigen::BiCGSTAB<SpMat> bicgstab;
    bicgstab.setMaxIterations(maxit);
    bicgstab.setTolerance(tol);
    x = bicgstab.compute(A).solve(b);

    std::cout << "--Eigen native BiCGSTAB" << std::endl;
    std::cout << "#iterations:       " << bicgstab.iterations() << std::endl;
    std::cout << "relative residual: " << bicgstab.error()      << std::endl;
    std::cout << "effective error:   " << (x-e).norm()    << std::endl;
    

    return 0;
}
