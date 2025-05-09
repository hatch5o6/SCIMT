import argparse
import os
from tqdm import tqdm

# Checks that everything in F1 has a copy in F2
# NOTE: Does not check the reverse!!!
parser = argparse.ArgumentParser()
# parser.add_argument("-f1", "--file1")
# parser.add_argument("-f2", "--file2")
parser.add_argument("-f1", "--folder1")
parser.add_argument("-f2", "--folder2")
parser.add_argument("-o", "--out")
args = parser.parse_args()

def do_the_thing():
    ALL_GOOD = True
    for f1 in os.listdir(args.folder1):
        f1 = os.path.join(args.folder1, f1)
        f2 = f1.replace(args.folder1, args.folder2)
        # print("-----------------------------------------")
        # print("F1:", f1)
        # print("F2:", f2)
        assert os.path.exists(f2)
        if not check(f1, f2):
            print(f"{f1} != {f2}")
            ALL_GOOD = False
    if ALL_GOOD:
        print(f"ALL FILES IN {args.folder1} HAVE EQUALS IN {args.folder2}")


def check(file1, file2):
    # print("reading", file1)
    with open(file1) as inf:
        f1_data = inf.read()

    # print("reading", file2)
    with open(file2) as inf:
        f2_data = inf.read()

    return f1_data == f2_data

if __name__ == "__main__":
    do_the_thing()