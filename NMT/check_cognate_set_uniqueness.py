import argparse

def check(f, tgt_file=None):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    
    if tgt_file:
        with open(tgt_file) as inf:
            tgt_lines = [l.strip() for l in inf.readlines()]
    else:
        tgt_lines = ["" for i in range(len(lines))]

    checked_lines = {}
    print("DUPLICATES:")
    for l, line in enumerate(lines):
        if line not in checked_lines:
            checked_lines[line] = []
        checked_lines[line].append(l)
    
    ct = 0
    for line, idxs in checked_lines.items():
        if len(idxs) > 1:
            ct += 1
            print(f"`{line}`:")
            for idx in idxs:
                print(f"\t{idx}: `{tgt_lines[idx]}`")
    print(f"{ct} WORDS OCCUR MORE THAN ONCE")
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    parser.add_argument("-t", "--tgt_file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    check(args.file, args.tgt_file)