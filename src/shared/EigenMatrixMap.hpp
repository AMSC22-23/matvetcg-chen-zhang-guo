#ifndef EIGEN_MATRIX_MAP_HPP
#define EIGEN_MATRIX_MAP_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <type_traits>

template<typename Scalar, std::size_t Rows, std::size_t Cols, typename MappedMatrix>
class EigenMatrixMap
{
public:
  //TODO: consider using cpp concempts for MappedMatrix type
  static EigenMatrixMap<Scalar, Rows, Cols, MappedMatrix>
  create_map(MappedMatrix const &m) {
    Scalar* data = const_cast<Scalar*>(m.data()); //const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar*>, "Mapping different scalar types");

    return EigenMatrixMap<Scalar, Rows, Cols, MappedMatrix>(data);
  }

  Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols>>&
  get_map() {
    return matrix_map;
  }

protected:
  EigenMatrixMap(Scalar* data) : matrix_map(data, Rows, Cols) {}

  Eigen::Map<Eigen::Matrix<Scalar, Rows, Cols>> matrix_map;
};

#endif //EIGEN_MATRIX_MAP_HPP
