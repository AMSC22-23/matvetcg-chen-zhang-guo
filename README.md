# matvetcg-chen-zhang-guo

A parallel MPI implementation of different iterative schemes.

## Supported Iterative schemes
- [x] Conjugate Gradient

## Supported preconditioners

## Compilation
```
- cd src
- mkdir build
- cd build
- cmake ..
- make [-j n]
```

## Run
```
./your_executable
```
or (if MPI used)
```
mpirun -n your_core_number your_executable
```


## File format
In order to maintain a consistent format please format your files with
```
clang-format -style=Google -i path/to/file
```
