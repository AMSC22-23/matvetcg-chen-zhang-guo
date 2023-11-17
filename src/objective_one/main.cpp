#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cg.hpp>
#include <iostream>

using std::cout;
using std::endl;

template <typename Lhs, typename Rhs, typename Scalar, int Size,
          typename ExactSol>
int cg_solve(Lhs &a, Rhs b, ExactSol &e) {
  // result vector
  apsc::LinearAlgebra::Vector<Scalar> x(Size, static_cast<Scalar>(0.0));
  auto max_iter = 10000;
  auto tol = 1e-8;

  // using no preconditioner
  apsc::LinearAlgebra::MatrixWithVecSupport<
      double, apsc::LinearAlgebra::ORDERING::ROWMAJOR>
      P(Size, Size);
  Scalar one = static_cast<Scalar>(1.0);
  for (unsigned i = 0; i < Size; i++) {
    P(i, i) = one;
  }

  auto result = LinearAlgebra::CG(a, x, b, P, max_iter, tol);

  cout << "Solution with (Squared) Conjugate Gradient:" << endl;
  cout << "iterations performed:                      " << max_iter << endl;
  cout << "tolerance achieved:                        " << tol << endl;
  cout << "Error norm:                                " << (x - e).norm()
       << std::endl;

  return result;
}

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr unsigned size = 10;
  cout << "Creating a test matrix..." << endl;
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> A(size,
                                                                          size);
  Utils::default_spd_fill<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,
                          double>(A);

  cout << "Creating a test rhs..." << endl;
  Vector<double> e(size, 1.0);
  Vector<double> b = A * e;

  // return cg_solve<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,

  return 0;
}
