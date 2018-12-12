# coding: utf-8
import sys
import subprocess
import multiprocessing
from time import time
from functools import partial

n_rep = 100
experiment_script = None
n_processes = 1


def launch_experiment(seed, args):
    print("Run task %i with args %s:" % (seed, " ".join(args)))
    start = time()
    print(" ".join(["python", experiment_script, "%i" % seed, *args]))
    try:
        subprocess.run(
            ["python", experiment_script, "%i" % seed, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        elapsed_time = time() - start
        print("Finished task %i, elapsed time: %f" % (seed, elapsed_time))
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise e


if __name__ == "__main__":
    try:
        experiment_script = sys.argv[1]
        n_processes = int(sys.argv[2])
        args = sys.argv[3:]
        print(
            "running experiment {expid} with {n_proc} process(es)".format(
                expid=experiment_script, n_proc=n_processes
            )
        )
    except:
        raise ValueError(
            "usage: "
            "python sim_loop.py [experiment script] [n_processes] [other args]"
        )
    seed_list = list(range(n_rep))
    launch_experiment_set = partial(launch_experiment, args=args)
    with multiprocessing.Pool(n_processes) as pool:
        r = pool.map(launch_experiment_set, seed_list, chunksize=1)
