import os

process_num = [1, 2, 4]
input_sizes = [100, 200, 300, 400, 500, 700, 1000, 2000, 3000, 4000, 8000, 9000, 10000, 11000, 12000, 15000, 20000]

build_dir = "../build/"

os.system(f'cd {build_dir} && cmake .. && make -j 4')

for p in process_num:
    for i in input_sizes:
        print(f'\nLaunching MPI with {p} process. Problem size: {i}')
        if os.system(f'cd {build_dir} && mpirun -n {p} objective2 {i}') != 0:
            print(f'Failure: MPI with {p} process. Problem size: {i}')
        else:
            print(f'Success: MPI with {p} process. Problem size: {i}\n')
