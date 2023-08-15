#!/usr/bin/python3
import multiprocessing as mp
import os

def trainer(script):
    print(f"Running {script}...")
    os.system(f"./{script}")

if __name__ == "__main__":
    scripts = ["train1.sh", "train2.sh"]
    with mp.Pool(processes=len(scripts)) as pool:
        pool.map(trainer, scripts)