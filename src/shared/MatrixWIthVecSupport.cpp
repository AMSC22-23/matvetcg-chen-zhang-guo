#include <Eigen/Dense>
#include <EigenStructureMap.hpp>
#include <Matrix/Matrix.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>

using namespace apsc::LinearAlgebra;

template <typename Scalar, ORDERING Order>
Vector<Scalar>
MatrixWithVecSupport<Scalar, Order>::solve(Vector<Scalar> const &v) {
  Vector<Scalar> x(MatrixWithVecSupport<Scalar, Order>::nRows,
                   static_cast<Scalar>(0.0));
  auto mat_map =
      EigenStructureMap<
          Eigen::Matrix<Scalar, MatrixWithVecSupport<Scalar, Order>::nRows,
                        MatrixWithVecSupport<Scalar, Order>::nCols>,
          Scalar, decltype(*this), MatrixWithVecSupport<Scalar, Order>::nRows,
          MatrixWithVecSupport<Scalar, Order>::nCols>::create_map(*this)
          .structure();

  Eigen::LDLT<decltype(mat_map)> spd_solver;
  return x;
}
