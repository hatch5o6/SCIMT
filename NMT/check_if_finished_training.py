import argparse
import os

FINISHED_TAG = """
################# (Lightning) #################
#######          TRAINING ENDED         #######
###############################################
""".strip()

def search(f, verbose):
    with open(f, errors="replace") as inf:
        content = inf.read()
    if FINISHED_TAG not in content:
        print(f"Didn't finish training:`{f}`")
    elif verbose:
        print(f"FINISHED TRAINING: `{f}`")

def all(d, verbose):
    for f in os.listdir(d):
        f_path = os.path.join(d, f)
        if os.path.isdir(f_path): continue
        if f_path.endswith(".xlsx"): continue
        slurm_job_id = f.split("_")[0]
        rest = f[len(slurm_job_id):]
        if rest.startswith("_TEST."): continue
        search(f_path, verbose)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates")
    parser.add_argument("--lang_pair", "-l")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    d = os.path.join(args.dir, args.lang_pair)
    all(d, args.verbose)
