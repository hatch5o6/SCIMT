import argparse
import os

def main(data_dir, langs):
    og_dirs = get_og_dirs(data_dir, langs)
    new_dirs = get_new_dirs(data_dir, og_dirs)
    print("og_dirs:", og_dirs)
    print("new_dirs:", new_dirs)
    for og, new in new_dirs.items():
        for new_dir in new:
            compare_dirs(og, new_dir, data_dir)

def compare_dirs(dir1, dir2, data_dir):
    print("\n\n\n################################################################")
    src1, tgt1 = dir1.split("-")
    src2, tgt2 = dir2.split("-")
    dir1_path = os.path.join(data_dir, dir1)
    dir2_path = os.path.join(data_dir, dir2)
    print(dir1_path)
    print(dir2_path, "\n")

    n_failed = 0
    for f1 in os.listdir(dir1_path):
        if ".SC_" in f1: continue

        f2 = None
        if f1.endswith(".notes"):
            f2 = f1
        elif f1.endswith(f"{src1}.txt"):
            # f2 = f1.replace(src1, src2)
            f2 = f1[:-len(f"{src1}.txt")] + f"{src2}.txt"
        elif f1.endswith(f"{tgt1}.txt"):
            # f2 = f1.replace(tgt1, tgt2)
            f2 = f1[:-len(f"{tgt1}.txt")] + f"{tgt2}.txt"
        else:
            raise ValueError(f"f1 is not proper: {f1}")
        assert f2 is not None
        
        f1_path = os.path.join(dir1_path, f1)
        f2_path = os.path.join(dir2_path, f2)

        passed = compare_files(f1_path, f2_path)
        if not passed:
            print("\tFAILED :(")
            n_failed += 1
        else:
            print("\tPASSED :)")
    
    print("FILES FAILED:", n_failed)

def compare_files(f1_path, f2_path):
    print(f"comparing: {f1_path}\n\t{f2_path}\n")
    f1_data = read_f(f1_path)
    f2_data = read_f(f2_path)
    return f1_data == f2_data
    
def get_og_dirs(data_dir, langs):
    dirs = []
    for d in os.listdir(data_dir):
        lang1, lang2 = d.split("-")
        if lang1 in langs and lang2 in langs:
            dirs.append(d)
    return dirs

def get_new_dirs(data_dir, og_dirs):
    new_dirs = {}
    for d in og_dirs:
        if d not in new_dirs:
            new_dirs[d] = []
        lang1, lang2 = d.split("-")
        lang1 = lang1[:2]
        lang2 = lang2[:2]
        for char in ["x", "y"]:
            possible_dir = f"{lang1}{char}-{lang2}{char}"
            if possible_dir in os.listdir(data_dir):
                new_dirs[d].append(possible_dir)
    return new_dirs

def read_f(f):
    with open(f) as inf:
        content = inf.read()
    return content

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data")
    parser.add_argument("--langs", default="es,an,en")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = set([l.strip() for l in args.langs.split(",")])
    if not len(langs) == 3:
        raise ValueError(f"more than 3 langs: {langs}")

    main(args.dir, langs)