/*
 * test_tools.cpp
 *
 *  Created on: Jan 3, 2024
 *      Author: Ying Zhang
 */

#include <cstddef>
#include <iostream>
#include <chrono>
#include <filesystem>

#include "utils.hpp"
#include "tools.hpp"
#include "MatrixWithVecSupport.hpp"

using std::cout;
using std::endl;


int main(int argc, char *argv[]) {
    using namespace apsc::LinearAlgebra;

    // Check if a filename is provided in the command line arguments
    if (argc != 2) {
        std::cout << "To run the program :\n";
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1; // Exit with an error code
    }

    std::string path = std::filesystem::current_path().string() + "/inputs/";
    path = path + argv[1];

    // Check if the file is opened successfully
    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl 
                  << "The mtx file should be placed in the **src/inputs** directory " << std::endl;
        return 1; // Exit with an error code
    }
    inputFile.close();


    std::cout << "Reading the matrix A..." << std::endl;
    MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> A;
    Tools::read_mtx_matrix<decltype(A)>(A, path);
    const unsigned size = A.rows();
    std::cout << "A has been loaded successfully" << std::endl;
    std::cout << "matrix A:\n" << A;

    std::cout << "Creating a spd matrix B..." << std::endl;
    const unsigned m = size;
    MatrixWithVecSupport<double, std::vector<double>, ORDERING::ROWMAJOR> B(m,m);
    // dense and random fill
    // B.fillRandom();
    // spd fill
    Utils::default_spd_fill<decltype(B), double>(B);
    std::cout << "matrix B:\n" << B;
    std::cout << "B has been created successfully" << std::endl;

    std::cout << "A * B = \n" << Tools::multiply_two_matrix<decltype(A)>(A,B) << std::endl;
    std::cout << "A - B = \n" << Tools::subtract_two_matrix<decltype(A)>(A,B) << std::endl;
    std::cout << "A + B = \n" << Tools::add_two_matrix<decltype(A)>(A,B) << std::endl;

    return 1;

}    
