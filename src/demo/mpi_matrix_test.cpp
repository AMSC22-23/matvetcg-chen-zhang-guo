#include "Vector.hpp"
#include <cstddef>
#include <cstdio>
#include <iostream>

#include <MPIMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <mpi.h>
#include <utils.hpp>

constexpr std::size_t size = 10;

int main(int argc, char* argv[]) {

  MPI_Init(nullptr, nullptr);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);
  
  //Create the global full matrix
  apsc::LinearAlgebra::MatrixWithVecSupport<
      double, apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      A(size, size);
  if (mpi_rank == 0) {
    apsc::LinearAlgebra::Utils::default_spd_fill<decltype(A), double>(A);
    std::cout << "Debug from rank: " << mpi_rank << std::endl << A << std::endl;
  }

  //Create a Vector
  apsc::LinearAlgebra::Vector<double> x(size, 1.0);

  //Create the MPI full matrix
  apsc::MPIMatrix<decltype(A), decltype(x)> PA;
  PA.setup(A, mpi_comm);
  int rank = 0;
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=";
      std::cout << PA.getLocalMatrix();
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }

  //Product
  PA.product(x);
  apsc::LinearAlgebra::Vector<double> res;
  PA.AllCollectGlobal(res);
  
  rank = 0;
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Res=";
      std::cout << res << std::endl;
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }

  MPI_Finalize();

  return 0;
}
