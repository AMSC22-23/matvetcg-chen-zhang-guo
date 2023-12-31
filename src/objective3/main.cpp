#include <mpi.h>
#include <stdint.h>

#include <Eigen/Sparse>
#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <MPISparseMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <iostream>
#include <utils.hpp>

#define DEBUG 0
#define USE_PRECONDITIONER 0
#define LOAD_MATRIX_FROM_FILE 1
#define ACCEPT_ONLY_SQUARE_MATRIX 1

using EigenVectord = Eigen::VectorXd;

constexpr uint8_t objective_id = 3;

int main(int argc, char *argv[]) {
#if LOAD_MATRIX_FROM_FILE == 0
  if (argc < 2) {
    std::cerr << "Please specify the problem size as input argument"
              << std::endl;
    return 0;
  }
  const int size = atoi(argv[1]);
#endif
  MPI_Init(&argc, &argv);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  Eigen::SparseMatrix<double, Eigen::ColMajor> A;
#if LOAD_MATRIX_FROM_FILE == 0
  if (mpi_rank == 0) {
    A.resize(size, size);
    for (int i = 0; i < size; i++) {
      A.insert(i, i) = 2.0;
      if (i > 0) {
        A.insert(i, i - 1) = -1.0;
      }
      if (i < size - 1) {
        A.insert(i, i + 1) = -1.0;
      }
    }
  }
#else
  if (argc < 2) {
    if (mpi_rank == 0) {
      std::cerr << "Input matrix not provided, terminating" << std::endl;
    }
    MPI_Finalize();
    return 0;
  }
  apsc::LinearAlgebra::Utils::EigenUtils::load_sparse_matrix<decltype(A),
                                                             double>(argv[1],
                                                                     A);
#if ACCEPT_ONLY_SQUARE_MATRIX == 1
  ASSERT(A.rows() == A.cols(),
         "The provided matrix is not square" << std::endl);
#endif
#endif
  std::cout << "Launching CG with a sparse MPI matrix with size: " << A.rows()
            << "x" << A.cols() << ", non zero: " << A.nonZeros() << std::endl;

  A.makeCompressed();

  // Maintain whole vectors in each processess
  EigenVectord e;
  EigenVectord b(A.rows());
  if (!mpi_rank) {
    e.resize(A.rows());
    e.fill(1.0);
    b = A * e;
#if DEBUG == 1
    std::cout << "e vector:" << std::endl << e << std::endl;
    std::cout << "b vector:" << std::endl << b << std::endl;
#endif
  }
  // Initialise processes b vector
  MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, mpi_comm);

  apsc::MPISparseMatrix<decltype(A), decltype(e),
                        decltype(A)::IsRowMajor
                            ? apsc::ORDERINGTYPE::ROWWISE
                            : apsc::ORDERINGTYPE::COLUMNWISE>
      PA;
  PA.setup(A, mpi_comm);
#if (DEBUG == 1)
  apsc::LinearAlgebra::Utils::MPI_matrix_show(PA, A, mpi_rank, mpi_size,
                                              mpi_comm);
#endif

#if USE_PRECONDITIONER == 0
  auto r = apsc::LinearAlgebra::Utils::conjugate_gradient::solve_MPI<
      decltype(PA), decltype(b), double, decltype(e)>(
      PA, b, e, MPIContext(mpi_comm, mpi_rank, mpi_size),
      objective_context(objective_id, mpi_size,
                        "objective" + std::to_string(objective_id) +
                            "_MPISIZE" + std::to_string(mpi_size) + ".log",
                        std::string(argv[1])));
#else
  // Setup the preconditioner, all the processes for now..
  // TODO
#endif
  MPI_Finalize();

  return r;
}
