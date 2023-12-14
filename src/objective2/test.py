import os

process_num = [1, 2, 4]
input_sizes = [100, 200, 500, 1000, 2000, 3000, 4000, 8000, 9000, 10000, 11000, 12000, 15000, 20000]

for p in process_num:
    print("With", p, "processes:")
    os.system(f'cd ../build && mpirun -n {p} objective2')
