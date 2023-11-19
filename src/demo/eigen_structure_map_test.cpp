#include <EigenStructureMap.hpp>
#include <MatrixWithVecSupport.hpp>
#include <Vector.hpp>
#include <cmath>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

int main(int argc, char *argv[]) {
  using namespace apsc::LinearAlgebra;

  constexpr unsigned size = 5;
  MatrixWithVecSupport<double, apsc::LinearAlgebra::ORDERING::ROWMAJOR> A(size,
                                                                          size);
  Utils::default_spd_fill<MatrixWithVecSupport<double, ORDERING::ROWMAJOR>,
                          double>(A);
  
  auto mapped_A = EigenStructureMap<Eigen::Matrix<double, size, size>, double, decltype(A), size, size>::create_map(A);
  cout << "Original matrix" << endl << A << endl;
  cout << "Mapped matrix" << endl << mapped_A.structure() << endl;
  mapped_A.structure() += mapped_A.structure();
  cout << "Modifying..." << endl;
  cout << "Original matrix" << endl << A << endl;
  cout << "Mapped matrix" << endl << mapped_A.structure() << endl;

  Vector<double> b(size, 1.0);
  auto mapped_b = EigenStructureMap<Eigen::Matrix<double, size, 1>, double, decltype(b), size>::create_map(b);
  cout << "Original vector" << endl << b << endl;
  cout << "Mapped vector" << endl << mapped_b.structure() << endl;
  mapped_b.structure() *= 2.0;
  cout << "Modifying..." << endl;
  cout << "Original vector" << endl << b << endl;
  cout << "Mapped vector" << endl << mapped_b.structure() << endl;

  return 0;
}
