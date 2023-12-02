#include <mpi.h>

#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cg_mpi.hpp>
#include <chrono>
#include <cstddef>
#include <iostream>

using std::cout;
using std::endl;

#define DEBUG 0
#define USE_PRECONDITIONER 0

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
  apsc::LinearAlgebra::Vector<Scalar> x(Size, static_cast<Scalar>(0.0));
  int max_iter = 2000;
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
      std::cout << "Process rank=" << mpi_rank << " Local Matrix=";
      std::cout << MPIMat.getLocalMatrix();
    }
    rank++;
    MPI_Barrier(mpi_comm);
  }
}

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  MPI_Init(&argc, &argv);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  constexpr unsigned size = 2000;

  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> A(size,
                                                                          size);
  if (mpi_rank == 0) {
    cout << "Launching CG with problem (SPD matrix) size of " << size << endl;
    Utils::default_spd_fill<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,
                            double>(A);
  }

  // Maintain whole vectors in each processes
  Vector<double> e(size, 1.0);
  Vector<double> b = A * e;
  // Initialise processes b vector
  MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, mpi_comm);

  apsc::MPIMatrix<decltype(A), decltype(b)> PA;
  PA.setup(A, mpi_comm);
#if (DEBUG == 1)
  MPI_matrix_show(PA, A, mpi_rank, mpi_size, mpi_comm);
#endif

#if USE_PRECONDITIONER == 0
  auto r = cg_solve_mpi<decltype(PA), decltype(b), double, size, decltype(e)>(
      PA, b, e, MPIContext(&mpi_comm, mpi_rank));
#else
  // Setup the preconditioner, all the processes for now..
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> P(size,
                                                                          size);
  for (unsigned i = 0; i < size; i++) {
    P(i, i) = 1.0;
  }
  MPI_Barrier(mpi_comm);
  auto r =
      cg_solve_mpi<decltype(PA), decltype(b), double, size, decltype(P),
                   decltype(e)>(PA, b, e, P, MPIContext(&mpi_comm, mpi_rank));
#endif
  MPI_Finalize();

  return r;
}
