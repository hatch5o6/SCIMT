import argparse
import os


 # JUST DO IT BY HAND. FORGET THIS.
def main(
    tok_folder
):
    for f in os.listdir(tok_folder):
        if "2" in f: continue
        f_path = os.listdir(tok_folder, f)
        if f == "GT":
            assert os.path.isdir(f_path)
            continue

        


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok_folder", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/tok")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.tok_folder)