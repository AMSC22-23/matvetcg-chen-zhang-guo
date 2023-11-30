import os

process_num = [1, 2, 4]

for p in process_num:
    print("With", p, "processes:")
    os.system(f'cd ../build && mpirun -n {p} objective2')
