#include <Eigen/Dense>
#include <EigenMatrixMap.hpp>
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
      EigenMatrixMap<Scalar, MatrixWithVecSupport<Scalar, Order>::nRows,
                     MatrixWithVecSupport<Scalar, Order>::nCols,
                     MatrixWithVecSupport<Scalar, Order>>::create_map(*this);
  return x;
}
