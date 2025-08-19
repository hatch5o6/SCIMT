import argparse
import csv
import os
import shutil

def get_data(
    train_csvs,
    val_csvs,
    test_csvs,
    out_dir,
    SC_MODEL_ID
):
    print("### READING ###")
    data = read_csvs(
        fs=train_csvs + val_csvs + test_csvs
    )
    print("### WRITING ###")
    write_data(data, out_dir, SC_MODEL_ID=SC_MODEL_ID)

def make_unique(lines):
    unique = set()
    final_lines = []
    for line in lines:
        if line in unique:
            pass
        else:
            final_lines.append(line)
            unique.add(line)
    return final_lines

def write_data(data, out_dir, SC_MODEL_ID):
    if os.path.exists(out_dir):
        print("deleting", out_dir)
        shutil.rmtree(out_dir)
    print("creating", out_dir)
    os.mkdir(out_dir)
    for lang, fs in data.items():
        print("--")
        print(f"Writing {lang}:")
        out_f = os.path.join(out_dir, f"{lang}.txt")
        lines = []
        for f in fs:
            if "{SC_MODEL_ID}" in f:
                print("FILE PATH REQUIRES SC_MODEL_ID")
                print("\t", f)
                assert SC_MODEL_ID is not None
                assert isinstance(SC_MODEL_ID, str)
                assert SC_MODEL_ID.lower() not in ["null", "none"]
                f = f.replace("{SC_MODEL_ID}", SC_MODEL_ID)
                print("\tINSERTING SC_MODEL_ID INTO PATH")
                print("\tFILE now is:", f)

            if not os.path.exists(f):
                print(f, "DOES NOT EXIST!")
            assert os.path.exists(f)
            print(f"\t-{f}")
            with open(f) as inf:
                lines += [line.strip() for line in inf.readlines()]
                print(f"\ttotal lines for {lang} now {len(lines)}")
        print("TOTAL LINES (BEFORE UNIQUE):", len(lines))
        # lines = make_unique(lines)
        print("TOTAL LINES (AFTER UNIQUE):", len(lines))
        print(f"Writing {lang} data ({len(lines)} lines) to {out_f}")
        with open(out_f, "w") as outf:
            outf.write("\n".join(lines) + "\n")

def read_csvs(fs):
    all_rows = []
    for csv_f in fs:
        if not os.path.exists(csv_f):
            print(csv_f, "DOES NOT EXIST!")
        print("CSV:", csv_f)
        assert os.path.exists(csv_f)
        print("Reading", csv_f)
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
    parser.add_argument("--train_csvs")
    parser.add_argument("--val_csvs")
    parser.add_argument("--test_csvs")
    parser.add_argument("--out_dir")
    parser.add_argument("--SC_MODEL_ID")
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
    train_csvs = [c.strip() for c in args.train_csvs.split(",")]
    val_csvs = [c.strip() for c in args.val_csvs.split(",")]
    test_csvs = [c.strip() for c in args.test_csvs.split(",")]
    get_data(
        train_csvs=train_csvs,
        val_csvs=val_csvs,
        test_csvs=test_csvs,
        out_dir=args.out_dir,
        SC_MODEL_ID=args.SC_MODEL_ID
    )