
/*
 * test_mpi.cpp
 *
 *  Created on: Dec 29, 2023
 *      Author: Ying Zhang
 */

#include <cstring>
#include <filesystem>
#include <cstddef>
#include <iostream>
#include <chrono>
#include "spai_mpi.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using SpMat=Eigen::SparseMatrix<double>;
using SpVec=Eigen::VectorXd;

using std::cout;
using std::endl;

static double c_start, c_diff;
#define tic() c_start = MPI_Wtime();
#define toc()                                                                \
  {                                                                          \
    c_diff = MPI_Wtime() - c_start;                                          \
    std::cout << "Time taken by SPAI_MPI: " << c_diff << " microseconds\n";  \
  }

int main(int argc, char *argv[]) {
    using namespace apsc::LinearAlgebra;

    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::string path;
    SpMat A;
    if (world_rank == 0) {
        // Check if parameters are provided in the command line arguments
        if (argc != 4) {
            std::cout << "To run the program :\n";
            std::cerr 
                << "Usage: " << argv[0] << " <filename> <max_iter> <epsilon>\n"
                << "<max_iter> is the maximal number of iterations to limit fill-in per column in M\n"
                << "<epsilon> is the stopping criterion on ||r||2 for every column of M" << std::endl;
            MPI_Finalize();          
            return 1; // Exit with an error code
        }
    }    
    MPI_Barrier(MPI_COMM_WORLD);
    path = std::filesystem::current_path().string() + "/inputs/" + argv[1];

    if (world_rank == 0) {    
        // std::cout << "Current path is " << std::filesystem::current_path() << "\n";
        // Check if the file is opened successfully
        std::ifstream inputFile(path);
        if (!inputFile.is_open()) {
            std::cerr << "Error opening file: " << path << "\n"
                    << "The mtx file should be placed in the ** src/inputs ** directory " << std::endl;
            MPI_Finalize();
            return 1; // Exit with an error code
        }
        inputFile.close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    Eigen::loadMarket(A, path);

    // Check A properties
    if (world_rank == 0) {
        // std::cout << "\nmatrix A:\n" << A << "\n"; 
        std::cout << "Matrix size:"<< A.rows() << " X " << A.cols() << std::endl;
        std::cout << "Non zero entries:" << A.nonZeros() << std::endl;
        SpMat B = SpMat(A.transpose()) - A;  // Check symmetry
        std::cout << "Norm of skew-symmetric part: " << B.norm() << std::endl;
    }
    const unsigned size = A.rows();

    // get M
    SpMat M(size, size);
    int max_iter; 
    double epsilon;
    if (world_rank == 0) {
        std::cout << "Creating the Matrix M(THE PRECONDITIONING OF A WITH SPARSE APPROXIMATE INVERSES)..." << std::endl;
        std::stringstream(argv[2]) >> max_iter;
        std::stringstream(argv[3]) >> epsilon;
    }
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&epsilon, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    tic();
    LinearAlgebra::SPAI_MPI<decltype(A), decltype(epsilon)>(A, M, max_iter, epsilon, MPI_COMM_WORLD);
    if (world_rank == 0) {
        toc();
    }

    // iterative solvers
    if (world_rank == 0) {
        // std::cout << "\nmatrix M:\n" << M << "\n";
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

    }
    
    MPI_Finalize();
    return 0;
}
