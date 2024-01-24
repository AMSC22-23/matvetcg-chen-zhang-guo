# matvetcg-chen-zhang-guo

## main result of yingzhang
Implement `SPAI` algorithm which is 
a parallel preconditioner based on approximate inverse  according to paper `ApproximateInversePreconditioner.pdf`.

## versions of `SPAI`
1. normal version without parallel - `spai.hpp` `test_spai.cpp`
2. openmp version parallelized by OpenMP based on normal version - `spai_openmp.hpp` `test_openmp.cpp` 
3. mpi version parallelized by MPI based on normal version - `spai_mpi.hpp` `test_mpi.cpp`
4. use `MatrixWithVecSupport` class as the base matrix (comes from Mattteochen's work: [link](https://github.com/AMSC22-23/matvetcg-chen-zhang-guo/blob/m5/src/shared/MatrixWithVecSupport.hpp) ) instead of sparse matrix of the Eigen library in the above work based on normal version - `spai_mbase.hpp` `test_mbase.cpp`

## others
To verify the results, I also create `spai_debug.hpp` which is only used for printing info of variables on the basis of `spai.hpp` to check whether each part meets expectation or not.

`test_baseline.cpp` is executed without preconditioner which is used as a baseline compared with different versions of `SPAI`.

`tools.hpp` is used to complement the missing parallel version of addition, subtraction, and multiplication between the two matrices in the `MatrixWithVecSupport` class. 

*.mtx files should be placed in the /src/inputs folder.


## Iterative schemes
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
./build/test_mbase <n> <max_iter> <epsilon>
```
For example
```
./build/test_mbase 6 10 0.6
```

5. baseline test
```
./build/test_baseline <filename>
```
For example 
```
./build/test_baseline test.mtx
```


