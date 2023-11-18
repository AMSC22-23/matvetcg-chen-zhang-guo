/*
 * EigenStructureMap.hpp
 *
 *  Created on: Nov 17, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cassert>

//Custom assertion supporting messages
#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "Assertion `" #condition "` failed in " << __FILE__         \
                << " line " << __LINE__ << ": " << message << std::endl;       \
      std::terminate();                                                        \
    }                                                                          \
  } while (false)

namespace apsc::LinearAlgebra {
namespace Utils {
template <typename Mat, typename Scalar> void default_spd_fill(Mat &m) {
  assert(m.rows() == m.cols() && "Must be a square matrix");
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
} // namespace Utils

} // namespace apsc::LinearAlgebra

#endif // UTILS_HPP
