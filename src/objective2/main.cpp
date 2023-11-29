#include "Matrix/Matrix.hpp"
#include "utils.hpp"
#include <cstddef>
#include <iostream>
#include <MatrixWithVecSupport.hpp>
#include <PMatrix.hpp>
#include <mpi.h>

constexpr std::size_t size = 10;

int main(int argc, char* argv[]) {

  MPI_Init(nullptr, nullptr);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  apsc::LinearAlgebra::MatrixWithVecSupport<
      double, apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      A(size, size);
  if (mpi_rank == 0) {
    apsc::LinearAlgebra::Utils::default_spd_fill<decltype(A), double>(A);
    std::cout << "Debug from rank: " << mpi_rank << std::endl << A << std::endl;
  }

  apsc::PMatrix<decltype(A)> PA;
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

  MPI_Finalize();

  return 0;
}
