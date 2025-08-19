import argparse
import os

def check(slurm_outputs_dir):
    u = 0
    print("UNFINISHED JOBS:")
    for f in os.listdir(slurm_outputs_dir):
        assert f.endswith(".out")
        f_path = os.path.join(slurm_outputs_dir, f)
        with open(f_path) as inf:
            lines = [l.rstrip() for l in inf.readlines()]
        if "Finished-----------------------" not in lines:
            print(f"\t{u}) `{f_path}`")
            u += 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--slurm_outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    check(args.slurm_outputs)