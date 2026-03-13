"""
Usage:
activate haddock3 environment
python3 3_launch_haddock_runs.py path_to_CFG_directory
"""

import sys, os
import subprocess


def sequential_launch(path_init):
    list_dirs = [os.path.join(path_init,d) for d in  os.listdir(path_init)]
    list_dirs = [d for d in list_dirs if os.path.exists(d)]
    print(f'Running {path_init} for {len(list_dirs)} directories')
    curr_dir = os.getcwd()
    for i,digit in enumerate(list_dirs):
        if os.path.exists(os.path.join(digit,'topology')):
            print('Skipping {digit}\n Already processed')
            continue
        os.chdir(digit)
        try:
            subprocess.run("haddock3 topology.cfg",shell=True)
            print(f'run success {os.path.basename(digit)} {i}')
        except:
            print(f'run failed {os.path.basename(digit)} {i}')
        os.chdir(curr_dir)
        
    print(f"\ndone {path_init}")
    return

path_runs = sys.argv[1] 
print(sys.argv)

sequential_launch(path_runs)
print('Finished !!')