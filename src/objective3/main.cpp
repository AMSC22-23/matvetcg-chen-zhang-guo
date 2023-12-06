#include <chrono>
#include <cstddef>
#include <iostream>

#include <Eigen/Sparse>
#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <MPISparseMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Parallel/Utilities/partitioner.hpp>
#include <Vector.hpp>
#include <cg_mpi.hpp>
#include <mpi.h>

#define DEBUG 0
#define USE_PRECONDITIONER 0

constexpr int size = 2000;

using std::cout;
using std::endl;

// From Eigen 3.4 you can use Eigen::VectorX<Scalar>
using EigenVectord = Eigen::Matrix<double, size, 1>;

#if USE_PRECONDITIONER == 0
template <typename MPILhs, typename Rhs, typename Scalar, int Size,
          typename ExactSol>
int cg_solve_mpi(MPILhs &A, Rhs b, ExactSol &e, const MPIContext mpi_ctx) {
#else
template <typename MPILhs, typename Rhs, typename Scalar, int Size,
          typename MPIPrecon, typename ExactSol>
int cg_solve_mpi(MPILhs &A, Rhs b, ExactSol &e, MPIPrecon /*P*/,
                 const MPIContext mpi_ctx) {
#endif
  Rhs x;
  x.resize(Size);
  x.fill(0.0);
  int max_iter = 5000;
  Scalar tol = 1e-18;
#if USE_PRECONDITIONER == 0
  std::chrono::high_resolution_clock::time_point begin =
      std::chrono::high_resolution_clock::now();
  auto result = LinearAlgebra::CG_no_precon<MPILhs, Rhs, Size, Scalar>(
      A, x, b, max_iter, tol, mpi_ctx, MPI_DOUBLE);
  std::chrono::high_resolution_clock::time_point end =
      std::chrono::high_resolution_clock::now();

  if (mpi_ctx.mpi_rank() == 0) {
    // We are assuming that each process takes more or less the same
    // computational time
    std::cout << "Elapsed time = "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       begin)
                     .count()
              << "[µs]" << std::endl;
    cout << "Solution with Conjugate Gradient:" << endl;
    cout << "iterations performed:                      " << max_iter << endl;
    cout << "tolerance achieved:                        " << tol << endl;
    cout << "Error norm:                                " << (x - e).norm()
         << std::endl;
#if DEBUG == 1
    cout << "Result vector:                             " << x << std::endl;
#endif
  }
  return result;
#else
  // TODO
#endif
}

template <typename MPIMatrix, typename Matrix>
void MPI_matrix_show(MPIMatrix MPIMat, Matrix Mat, const int mpi_rank,
                     const int mpi_size, MPI_Comm mpi_comm) {
  int rank = 0;
  while (rank < mpi_size) {
    if (mpi_rank == rank) {
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=" << std::endl;
      std::cout << MPIMat.getLocalMatrix();
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  Eigen::SparseMatrix<double, Eigen::ColMajor> A;
  if (mpi_rank == 0) {
    std::cout << "Launching CG with a sparse MPI matrix with global size of "
              << size << std::endl;
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

  A.makeCompressed();

  // Maintain whole vectors in each processess
  EigenVectord e;
  EigenVectord b;
  if (!mpi_rank) {
    e.resize(size);
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
  MPI_matrix_show(PA, A, mpi_rank, mpi_size, mpi_comm);
#endif

#if USE_PRECONDITIONER == 0
  auto r = cg_solve_mpi<decltype(PA), decltype(b), double, size, decltype(e)>(
      PA, b, e, MPIContext(&mpi_comm, mpi_rank));
#else
  // Setup the preconditioner, all the processes for now..
  // TODO
#endif
  MPI_Finalize();

  return r;
}
