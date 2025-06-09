import argparse
import csv
import os
import shutil

def get_data(
    train_csv,
    val_csv,
    test_csv,
    out_dir
):
    data = read_csvs(
        fs=[train_csv, val_csv, test_csv]
    )
    write_data(data, out_dir)

def write_data(data, out_dir):
    if os.path.exists(out_dir):
        print("deleting", out_dir)
        shutil.rmtree(out_dir)
    print("creating", out_dir)
    os.mkdir(out_dir)
    for lang, fs in data.items():
        out_f = os.path.join(out_dir, f"{lang}.txt")
        lines = []
        for f in fs:
            with open(f) as inf:
                lines += [line.strip() for line in inf.readlines()]
        print(f"Writing {lang} data to {out_f}")
        with open(out_f, "w") as outf:
            outf.write("\n".join(lines) + "\n")

def read_csvs(fs):
    all_rows = []
    for csv_f in fs:
        with open(csv_f, newline="") as inf:
            rows = [row for row in csv.reader(inf)]
        header = rows[0]
        rows = [tuple(row) for row in rows[1:]]
        all_rows += rows
    
    data = get_lang_paths(all_rows)
    return data

def get_lang_paths(rows):
    data = {}
    for src_lang, tgt_lang, src_f, tgt_f in rows:
        if src_lang not in data:
            data[src_lang] = []
        if tgt_lang not in data:
            data[tgt_lang] = []

        if src_f not in data[src_lang]:
            data[src_lang].append(src_f)
        if tgt_f not in data[tgt_lang]:
            data[tgt_lang].append(tgt_f)
    return data
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv")
    parser.add_argument("--val_csv")
    parser.add_argument("--test_csv")
    parser.add_argument("--out_dir")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("############################")
    print("# make_SC_training_data.py #")
    print("############################")
    args = get_args()
    get_data(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        out_dir=args.out_dir
    )