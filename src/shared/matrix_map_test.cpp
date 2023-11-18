#include "EigenMatrixMap.hpp"
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cmath>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr unsigned size = 10;
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> A(size,
                                                                          size);
  Utils::default_spd_fill<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,
                          double>(A);
  
  auto mapped_A = EigenMatrixMap<double, size, size, MatrixWithVecSupport<double>>::create_map(A);
  cout << "Original matrix" << endl << A << endl;
  cout << "Mapped matrix" << endl << mapped_A.get_map() << endl;

  return 0;
}
