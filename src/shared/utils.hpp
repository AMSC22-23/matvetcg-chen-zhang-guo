/*
 * EigenStructureMap.hpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <mpi.h>

#include <assert.hpp>
#include <cassert>
#include <cg_mpi.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <unsupported/Eigen/SparseExtra>

using std::cout;
using std::endl;

#define PRODUCE_OUT_FILE 1
#define CG_MAX_ITER(i) (20 * i)

namespace apsc::LinearAlgebra {
namespace Utils {
template <typename Mat, typename Scalar>
void default_spd_fill(Mat &m) {
  ASSERT((m.rows() == m.cols()), "The matrix must be squared!");
  const Scalar diagonal_value = static_cast<Scalar>(2.0);
  const Scalar upper_diagonal_value = static_cast<Scalar>(-1.0);
  const Scalar lower_diagonal_value = static_cast<Scalar>(-1.0);
  const auto size = m.rows();

  for (unsigned i = 0; i < size; ++i) {
    m(i, i) = diagonal_value;
    if (i > 0) {
      m(i, i - 1) = upper_diagonal_value;
    }
    if (i < size - 1) {
      m(i, i + 1) = lower_diagonal_value;
    }
  }
}

namespace EigenUtils {
template <typename Matrix, typename Scalar>
void load_sparse_matrix(const std::string file_name, Matrix &mat) {
  static_assert(
      std::is_same_v<Matrix, Eigen::SparseMatrix<Scalar, Eigen::RowMajor>> ||
          std::is_same_v<Matrix, Eigen::SparseMatrix<Scalar, Eigen::ColMajor>>,
      "Matrix type must be an Eigen::SparseMatrix of type Scalar");
  ASSERT(Eigen::loadMarket(mat, file_name),
         "Failed to load matrix from " << file_name);
}
}  // namespace EigenUtils

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

namespace conjugate_gradient {
template <typename MPILhs, typename Rhs, typename Scalar, typename ExactSol,
          int ObjectiveNumber, typename... Preconditioner>
int solve_MPI(MPILhs &A, Rhs b, ExactSol &e, const MPIContext mpi_ctx,
              Preconditioner... P) {
  constexpr std::size_t P_size = sizeof...(P);
  static_assert(P_size < 2, "Please specify max 1 preconditioner");

  const int size = b.size();

  Rhs x;
  x.resize(size);
  x.fill(0.0);
  int max_iter = CG_MAX_ITER(size);
  Scalar tol = 1e-10;

  if constexpr (P_size == 0) {
    std::chrono::high_resolution_clock::time_point begin =
        std::chrono::high_resolution_clock::now();
    auto result = ::LinearAlgebra::CG_no_precon<MPILhs, Rhs, Scalar>(
        A, x, b, max_iter, tol, mpi_ctx, MPI_DOUBLE);
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    long long diff =
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();
    decltype(diff) diff_sum = 0;

    MPI_Reduce(&diff, &diff_sum, 1, MPI_LONG_LONG, MPI_SUM, 0,
               mpi_ctx.mpi_comm());

    if (mpi_ctx.mpi_rank() == 0) {
      cout << "(Time spent by all processes: " << diff_sum
           << ", total processes: " << mpi_ctx.mpi_size() << ")" << std::endl
           << std::endl;
      cout << "Mean elapsed time = " << diff_sum / mpi_ctx.mpi_size() << "[µs]"
           << endl;
      cout << "Solution with Conjugate Gradient:" << endl;
      cout << "iterations performed:                      " << max_iter << endl;
      cout << "tolerance achieved:                        " << tol << endl;
      cout << "Error norm:                                " << (x - e).norm()
           << std::endl;
#if PRODUCE_OUT_FILE == 1
      {
        const std::string output_file_name =
            "objective" + std::to_string(ObjectiveNumber) + "_MPISIZE" +
            std::to_string(mpi_ctx.mpi_size()) + ".log";
        std::ofstream output_file(output_file_name, std::ios::app);
        if (output_file.is_open()) {
          output_file << size << " " << diff_sum / mpi_ctx.mpi_size()
                      << std::endl;
        } else {
          std::cerr << "Failed to open output file" << endl;
        }
        output_file.close();
      }
#endif
#if DEBUG == 1
      cout << "Result vector:                             " << x << std::endl;
#endif
    }
    return result;
  } else {
    // TODO
  }
}
}  // namespace conjugate_gradient
}  // namespace Utils
}  // namespace apsc::LinearAlgebra

#endif /*UTILS_HPP*/
