#include <cstddef>
#include <iostream>
#include <chrono>

#include <MPIContext.hpp>
#include <MPIMatrix.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cg_mpi.hpp>
#include <mpi.h>

using std::cout;
using std::endl;

#define DEBUG 0

//When called with MPI > 1 the `pa` is a submatrix of the original matrix
template <typename MPILhs, typename Rhs, typename Scalar,
          int Size, typename MPIPrecon, typename ExactSol>
int cg_solve_mpi(MPILhs &A, Rhs b, ExactSol &e, MPIPrecon P, const MPIContext mpi_ctx) {
  // result vector
  apsc::LinearAlgebra::Vector<Scalar> x(Size, static_cast<Scalar>(0.0));
  int max_iter = 10000;
  Scalar tol = 1e-18;
  
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  auto result = LinearAlgebra::CG<MPILhs, Rhs, MPIPrecon, Size, Scalar>(
      A, x, b, P, max_iter, tol, mpi_ctx, MPI_DOUBLE);
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << "rank = " << mpi_ctx.mpi_rank() << " CG time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
  
  if (mpi_ctx.mpi_rank() == 0) {
    cout << "Solution with Conjugate Gradient:" << endl;
    cout << "iterations performed:                      " << max_iter << endl;
    cout << "tolerance achieved:                        " << tol << endl;
    cout << "Error norm:                                " << (x - e).norm() << std::endl;
  }

  return result;
}

template<typename MPIMatrix, typename Matrix>
void MPI_matrix_show(MPIMatrix MPIMat, Matrix Mat, const int mpi_rank, const int mpi_size, MPI_Comm mpi_comm) {
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

  MPI_Init(nullptr, nullptr);
  int mpi_rank;
  int mpi_size;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(mpi_comm, &mpi_rank);
  MPI_Comm_size(mpi_comm, &mpi_size);

  constexpr unsigned size = 1000;
  
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> A(
      size, size);
  if (mpi_rank == 0) {
    cout << "Creating a test matrix..." << endl;
    Utils::default_spd_fill<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,
                            double>(A);
  }
  
  //Maintain whole vectors in each processes
  Vector<double> e(size, 1.0);
  Vector<double> b = A * e;
  //Initialise processes b vector
  MPI_Bcast(b.data(), b.size(), MPI_DOUBLE, 0, mpi_comm);

  apsc::MPIMatrix<decltype(A), decltype(b)> PA;
  PA.setup(A, mpi_comm);
#if (DEBUG == 1)
  MPI_matrix_show(PA, A, mpi_rank, mpi_size, mpi_comm);
#endif
  
  //Setup the preconditioner, all the processes for now..
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> P(size, size);
  for (unsigned i=0; i<size; i++) {
    P(i, i) = 1.0;
  }
  MPI_Barrier(mpi_comm);

  auto r = cg_solve_mpi<decltype(PA), decltype(b), double, size, decltype(P),
                  decltype(e)>(PA, b, e, P, MPIContext(&mpi_comm, mpi_rank));
  MPI_Finalize();

  return r;
}
