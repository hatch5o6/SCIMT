import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-f1", "--file1")
parser.add_argument("-f2", "--file2")
parser.add_argument("-o", "--out", default="diff_check.default.out")
parser.add_argument("--PRINT", action="store_true")
args = parser.parse_args()

print("reading", args.file1)
with open(args.file1) as inf:
    lines1 = [line.strip() for line in tqdm(inf.readlines())]

print("reading", args.file2)
with open(args.file2) as inf:
    lines2 = [line.strip() for line in tqdm(inf.readlines())]

assert len(lines1) == len(lines2)

pairs = list(zip(lines1, lines2))

print("comparing")
n_diffs = 0
with open(args.out, 'w') as outf:
    outf.write("NOT EQUAL:-\n")
    for p, (line1, line2) in tqdm(enumerate(pairs), total=len(pairs)):
        if line1 != line2:
            n_diffs += 1
            outf.write(f"----------------line {p + 1} ({n_diffs})-----------------\n")
            outf.write(line1 + "\n")
            outf.write(line2 + "\n")

            if args.PRINT:
                print(f"----------------line {p + 1} ({n_diffs})-----------------")
                print(line1)
                print(line2)

    outf.write(f"{n_diffs} DIFFERENT LINES")

print(f"{n_diffs} DIFFERENT LINES")