/*
 * tools.hpp
 *
 *  Created on: Jan 3, 2024
 *      Author: Ying Zhang
 */

#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <cassert>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <type_traits>
#include <cstring>
#include <fstream>
#include <sstream>

#include <assert.hpp>
#include "Matrix/Matrix.hpp"
#include <omp.h>
#include "assert.hpp"

namespace apsc::LinearAlgebra {
namespace Tools {
    /*!
     *  C = A * B in parallel with OpenMP
     */
    template<typename Mat>
    Mat multiply_two_matrix(Mat &A, Mat &B) {
        static_assert(
            std::is_base_of<Matrix<double, ORDERING::ROWMAJOR>, Mat>::value ||
            std::is_base_of<Matrix<double, ORDERING::COLUMNMAJOR>, Mat>::value,
            "This function is only used for Matrix class and its derived class.");
        std::size_t AROWS = A.rows();
        std::size_t ACOLS = A.cols();
        std::size_t BROWS = B.rows();
        std::size_t BCOLS = B.cols();

        ASSERT(ACOLS==BROWS, 
                    "the cols of the first matrix should be equal to the rows of the second matrix");
        Mat C(AROWS, BCOLS);
        #pragma omp parallel for 
        for (std::size_t i = 0; i < AROWS; i++) 
        {
            #pragma omp parallel for 
            for (std::size_t j = 0; j < BCOLS; j++) 
            {
                for (std::size_t z = 0; z < ACOLS; z++) 
                {
                    C(i,j) += A(i,z) * B(z,j);
                }
            }
        }

        return C;
    }

    /*!
     *  C = A - B in parallel with OpenMP
     */
    template<typename Mat>
    Mat subtract_two_matrix(Mat &A, Mat &B) {
        static_assert(
            std::is_base_of<Matrix<double, ORDERING::ROWMAJOR>, Mat>::value ||
            std::is_base_of<Matrix<double, ORDERING::COLUMNMAJOR>, Mat>::value,
            "This function is only used for Matrix class and its derived class.");
        std::size_t AROWS = A.rows();
        std::size_t ACOLS = A.cols();
        std::size_t BROWS = B.rows();
        std::size_t BCOLS = B.cols();

        ASSERT(AROWS==BROWS && ACOLS==BCOLS, 
                    "the size of the first matrix should be equal to the size of the second matrix");
        Mat C(AROWS, ACOLS);
        #pragma omp parallel for 
        for (std::size_t i = 0; i < AROWS; i++) 
        {
            #pragma omp parallel for 
            for (std::size_t j = 0; j < ACOLS; j++) 
            {
                C(i,j) = A(i,j) - B(i,j);
            }
        }

        return C;
    }

    /*!
     *  C = A + B in parallel with OpenMP
     */
    template<typename Mat>
    Mat add_two_matrix(Mat &A, Mat &B) {
        static_assert(
            std::is_base_of<Matrix<double, ORDERING::ROWMAJOR>, Mat>::value ||
            std::is_base_of<Matrix<double, ORDERING::COLUMNMAJOR>, Mat>::value,
            "This function is only used for Matrix class and its derived class.");
        std::size_t AROWS = A.rows();
        std::size_t ACOLS = A.cols();
        std::size_t BROWS = B.rows();
        std::size_t BCOLS = B.cols();

        ASSERT(AROWS==BROWS && ACOLS==BCOLS, 
                    "the size of the first matrix should be equal to the size of the second matrix");
        Mat C(AROWS, ACOLS);
        #pragma omp parallel for 
        for (std::size_t i = 0; i < AROWS; i++) 
        {
            #pragma omp parallel for 
            for (std::size_t j = 0; j < ACOLS; j++) 
            {
                C(i,j) = A(i,j) + B(i,j);
            }
        }

        return C;
    }

    /*！
     * read matrix from .mtx file
     */
    template<typename Mat>
    void read_mtx_matrix(Mat &A, std::string path) {
        std::ifstream file(path);
        std::string line;
        std::getline(file, line);
        std::getline(file, line); 

        int rows, cols, nonZeros;
        file >> rows >> cols >> nonZeros;
        // std::cout<<"rows="<<rows<<"  cols="<<cols<<" nonZeros="<<nonZeros<<"\n";
        A.resize(rows, cols);
        for (unsigned i = 0; i < rows; i++) {
            for (unsigned j = 0; i < cols; i++) {
                A(i,j) = 0.0;
            }
        }

        while (std::getline(file, line)) {
            int i, j;
            double value;
            file >> i >> j >> value;
            A(i-1, j-1) = value;
        }
        file.close();
    }

} // namespace Tools
} // namespace apsc::LinearAlgebra
#endif /*TOOLS_HPP*/

