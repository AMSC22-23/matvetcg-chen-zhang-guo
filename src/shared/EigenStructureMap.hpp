/*
 * EigenStructureMap.hpp
 *
 *  Created on: Nov 18, 2023
 *      Author: Kaixi Matteo Chen
 */

#ifndef EIGEN_MATRIX_MAP_HPP
#define EIGEN_MATRIX_MAP_HPP

#include <Eigen/Dense>
#include <cstddef>
#include <type_traits>

/*!
 * A full Eigen compatible matrix class with custom handled buffer data.
 * This class is meant to call Eigen methods on data now owned by Eigen, in
 * order to avoid memory movements. Refer to
 * http://www.eigen.tuxfamily.org/dox/group__TutorialMapClass.html
 *
 * @tparam EigenStructure The mapped Eigen type (MatrixX<>, VectorX<>, ...)
 * @tparam Scalar The scalar type
 * @tparam MappedMatrix The custom matrix type who owns the data buffer
 * @tparam Sizes The mapped Eigen type size
 */
template <typename EigenStructure, typename Scalar, typename MappedMatrix,
          std::size_t... Sizes>
class EigenStructureMap {
public:
  EigenStructureMap() = default;

  // TODO: consider using cpp concempts for MappedMatrix type
  static EigenStructureMap<EigenStructure, Scalar, MappedMatrix, Sizes...>
  create_map(MappedMatrix const &m) {
    Scalar *data =
        const_cast<Scalar *>(m.data()); // const versioon is called, why?

    static_assert(std::is_same_v<decltype(data), Scalar *>,
                  "Mapping different scalar types");
    return EigenStructureMap<EigenStructure, Scalar, MappedMatrix, Sizes...>(
        data);
  }

  auto structure() { return structure_map; }

protected:
  EigenStructureMap(Scalar *data) : structure_map(data, Sizes...) {}

  Eigen::Map<EigenStructure> structure_map;
};

#endif // EIGEN_MATRIX_MAP_HPP
