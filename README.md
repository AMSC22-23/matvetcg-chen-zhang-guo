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

## Run `objective1`
```
./your_executable
```

## Run `objective2`
```
mpirun -n [your_core_number] objective2 [problem_size]
```
For example:
```
mpirun -n 2 objective2 1000
```

## Run `objective3`
```
mpirun -n [your_core_number] objective3 [input_file]
```
For example (launched from `build` directory):
```
mpirun -n 2 objective3 ../inputs/0_05fill_size9604.mtx 
```

## File format
In order to maintain a consistent format please format your files with
```
clang-format -style=Google --sort-includes -i path/to/file
```
