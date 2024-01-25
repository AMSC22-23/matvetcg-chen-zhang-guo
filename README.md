# matvetcg-chen-zhang-guo
- The files mentioned below are placed in the folder /src/objective5 by default on branch M5 for this project.

- *.mtx files should be placed in the /src/inputs folder.

## main results of yingzhang
Implement `SPAI` algorithm which is 
a parallel preconditioner based on approximate inverse  according to paper `ApproximateInversePreconditioner.pdf`.

For details of specific yingzhang's work, please refer to the following two sections.

## Implemented four versions of `SPAI`
1. normal version without parallel - `spai.hpp` `test_spai.cpp`
2. openmp version parallelized by OpenMP based on normal version - `spai_openmp.hpp` `test_openmp.cpp` 
3. mpi version parallelized by MPI based on normal version - `spai_mpi.hpp` `test_mpi.cpp`
4. use `MatrixWithVecSupport` class as the base matrix (comes from my teammate Mattteochen's work: [link](https://github.com/AMSC22-23/matvetcg-chen-zhang-guo/blob/m5/src/shared/MatrixWithVecSupport.hpp) ) instead of sparse matrix of the Eigen library in the above work based on normal version - `spai_mbase.hpp` `test_mbase.cpp`
5. To compare the above four versions, I also created `test_baseline.cpp` which is executed without preconditioner and can be used as a baseline.

## Additional work
`spai_debug.hpp` - is only used for printing info of variables on the basis of `spai.hpp` to check whether each part meets expectation or not and the correctness of this algorithm.

`tools.hpp` - is used to complement the missing parallel version of addition, subtraction, and multiplication between two matrices in the `MatrixWithVecSupport` class. 

`src/algorithms/bicgstab.hpp` - I changed part of [BiCGSTAB](https://github.com/mattteochen/AMSC-CodeExamples/blob/c8f2e13b22d3dee4884b49ec497c49d5206044f0/Examples/src/LinearAlgebra/IML_Eigen/include/bicgstab.hpp) to make it compatible with our project. (The same changes to `src/algorithms/cg.hpp` came from my teammate Mattteochen.) These two iterative solvers are used to solve linear equation in `test_mbase.cpp` as substitutes of their counterpart in Eigen library.

## Iterative solvers
- [x] Conjugate Gradient
- [x] BiCGSTAB

## Setup
Initialise submodules

```
git submodule update --init --recursive
```

## Compilation
This repository is intended to be used inside the `pcafrica/mk` docker image
env. See
[here](https://github.com/HPC-Courses/AMSC-Labs/tree/main/Labs/2023-24/lab00-setup)
for the configuration. If so, you don't need to take additional steps.

In case you want to run the project locally, you have to:
- Export the `Eigen` include directory path under the env variable
  `$EIGEN3_INCLUDE_DIR`.
- Have `MPI` installed (Note: we had no issued using version `3.1` of `MPI`,
  newer versions might not compatible)
- Have `OpenMP` installed (version `4.5` of `OpenMP`)

Then:

```
- cd src
- ./test.sh
```

## Run `objective5`
1. normal version
```
./build/test_spai <filename> <max_iter> <epsilon>
```
For example
```
./build/test_spai test.mtx 10 0.6
```

2. openmp version
```
./build/test_openmp <filename> <max_iter> <epsilon>
```
For example
```
./build/test_openmp test.mtx 10 0.6
```

3. mpi version
```
mpirun -n <number_of_thread> ./build/test_mpi <filename> <max_iter> <epsilon>
```
For example
```
mpirun -n 4 ./build/test_mpi test.mtx 10 0.6
```

4. openmp version with `MatrixWithVecSupport` class
```
./build/test_mbase <filename> <max_iter> <epsilon>
```
For example
```
./build/test_mbase test.mtx 10 0.6
```

5. baseline test
```
./build/test_baseline <filename>
```
For example 
```
./build/test_baseline test.mtx
```


