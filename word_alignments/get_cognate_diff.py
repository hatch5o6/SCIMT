import argparse

def get_diff(f1, f2, out1, out2):
    with open(f1) as inf:
        f1_lines = set([
            line.strip() for line in inf.readlines()
        ])
    with open(f2) as inf:
        f2_lines = set([
            line.strip() for line in inf.readlines()
        ])
    
    f1_diff = f1_lines.difference(f2_lines)
    write_set(f1_diff, out1)

    f2_diff = f2_lines.difference(f1_lines)
    write_set(f2_diff, out2)

def write_set(set_a, fpath):
    with open(fpath, "w") as outf:
        for line in set_a:
            outf.write(line.strip() + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c1", "--cog1", required=True)
    parser.add_argument("-c2", "--cog2", required=True)
    parser.add_argument("--out1")
    parser.add_argument("--out2")
    return parser.parse_args()

def insert_ext(fname, new_ext):
    EXT = fname.split(".")[-1]
    return ".".join(fname.split(".")[:-1]) + f".{new_ext}.{EXT}"

if __name__ == "__main__":
    args = get_args()
    out1, out2 = args.out1, args.out2
    if not out1:
        out1 = insert_ext(args.cog1, "diff1")
    if not out2:
        out2 = insert_ext(args.cog2, "diff2")
    get_diff(args.cog1, args.cog2, out1, out2)