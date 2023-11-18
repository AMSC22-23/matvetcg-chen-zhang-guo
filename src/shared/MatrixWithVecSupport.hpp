/*
 * MatrixWithVecSupport.hpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef MATRIX_WITH_VEC_SUPPORT_HPP
#define MATRIX_WITH_VEC_SUPPORT_HPP
#include "utils.hpp"
#include <Matrix/Matrix.hpp>
#include <Vector.hpp>
#include <cassert>
// To avoid stupid warnings if I do not use openmp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
namespace apsc::LinearAlgebra {
/*!
 * A full matrix with vector multiplication support for
 * apsc::LinearAlgebra::Vector
 * @tparam Scalar The type of the element
 * @tparam ORDER The Storage order (default row-wise)
 */
template <typename SCALAR, ORDERING ORDER = ORDERING::ROWMAJOR>
class MatrixWithVecSupport : public Matrix<SCALAR, ORDER> {
public:
  using Scalar = SCALAR;

  /*!
   * Constructor may take number of rows and columns
   * @param nrows Number of rows
   * @param ncols Number of columns
   */
  MatrixWithVecSupport(std::size_t nrows = 0, std::size_t ncols = 0)
      : Matrix<SCALAR, ORDER>(nrows, ncols) {}

  /*!
   * Multiplication with a apsc::LinearAlgebra::Vector
   * The output vector has the matrix scalar type.
   *
   * @param v a apsc::LinearAlgebra::Vector vector
   * @return The result of A*v
   */
  template <typename InputVectorScalar>
  Vector<Scalar> operator*(Vector<InputVectorScalar> const &v) const {

    ASSERT((Matrix<Scalar, ORDER>::nCols == v.size()),
           "MatVetMul: Matrix columns != Vector rows");

    Vector<Scalar> res(Matrix<Scalar, ORDER>::nRows, static_cast<Scalar>(0));

    if constexpr (ORDER == ORDERING::ROWMAJOR) {
      // loop over rows
      for (std::size_t i = 0; i < Matrix<Scalar, ORDER>::nRows; ++i) {
        Scalar r{0};
        auto offset = i * Matrix<Scalar, ORDER>::nCols;
        // loop over columns (can be multithreaded)
#pragma omp parallel for shared(i, res, offset) reduction(+ : r)
        for (std::size_t j = 0; j < Matrix<Scalar, ORDER>::nCols; ++j) {
          r += Matrix<Scalar, ORDER>::buffer[offset + j] * v[j];
        }
        res[i] = r;
      }
    } else {
      // loop over the columns
      for (std::size_t j = 0; j < Matrix<Scalar, ORDER>::nCols; ++j) {
        auto c = v[j];
        auto offset = j * Matrix<Scalar, ORDER>::nRows;
        // loop over rows (can be multithreaded)
#pragma omp parallel for shared(j, res, c, offset)
        for (std::size_t i = 0; i < Matrix<Scalar, ORDER>::nRows; ++i) {
          res[i] += c * Matrix<Scalar, ORDER>::buffer[offset + i];
        }
      }
    }
    return res;
  }

  Vector<Scalar> solve(Vector<Scalar> const& b);
};

} // namespace apsc::LinearAlgebra
#pragma GCC diagnostic pop

#endif /* MATRIX_WITH_VEC_SUPPORT_HPP */
