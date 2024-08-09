import os
import subprocess
import glob
import argparse


# start this in a tmux window, will run all folders in parallel using srun
# don't forget: conda activate ...
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_identifier", type=str)
    args = parser.parse_args()
    run_identifier = args.run_identifier
    RES_DIRECTORIES = sorted(glob.iglob("/scratch/vito/scanpathes/results/*"))[::-1]
    grid_dirs = [x for x in RES_DIRECTORIES if run_identifier in x]
    # assert len(grid_dirs) == 98, "incorrect filter?!?"
    commands = [
        f'srun --time=0-01:00 --ntasks=1 --partition=c0,c1a,c1b,c2 --cpus-per-task=4 python /scratch/vito/scanpathes/code/domip_scanpathes/src/summary_plots_hpc.py --experiment_dir {RES_DIR} {"--eval_only_3sec" if "_3SEC_" in RES_DIR else ""}'
        for RES_DIR in grid_dirs
    ]
    print(commands)
    # create a list to store the subprocesses
    processes = []
    # start each srun command in a separate process
    for cmd in commands:
        processes.append(subprocess.Popen(cmd.split()))
    # wait for all processes to complete
    for proc in processes:
        proc.wait()


if __name__ == "__main__":
    main()
