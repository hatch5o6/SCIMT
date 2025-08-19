import argparse
import os
import random

def format(
    src_data_fs,
    tgt_data_fs,
    src,
    tgt,
    out_dir,
    pref,
    seed,
    EXCLUDE_SRC,
    EXCLUDE_TGT
):
    random.seed(seed)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    out_dir = os.path.join(out_dir, str(seed))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    og_src_data = read_data(src_data_fs)
    og_tgt_data = read_data(tgt_data_fs)
    assert len(og_src_data) == len(og_tgt_data)
    og_data = list(zip(og_src_data, og_tgt_data))
    random.shuffle(og_data)
    print("BEFORE UNIQUE:", len(og_data))
    og_data = make_unique(og_data)
    print("AFTER UNIQUE:", len(og_data))

    exclude_src_data = read_data(EXCLUDE_SRC)
    exclude_tgt_data = read_data(EXCLUDE_TGT)
    assert len(exclude_src_data) == len(exclude_tgt_data)
    exclude_data = set(list(zip(exclude_src_data, exclude_tgt_data)))
    og_data = remove_data(og_data)
    print("AFTER REMOVE EXCLUDE DATA:", len(og_data))
    
    src_data = []
    tgt_data = []
    for src_word, tgt_word in og_data:
        src_word = src_word.replace(" ", "_")
        tgt_word = tgt_word.replace(" ", "_")
        src_data.append(" ".join(src_word))
        tgt_data.append(" ".join(tgt_word))

    src_out_f = os.path.join(out_dir, f"{pref}_{src}_{tgt}.{src}")
    write_file(src_data, src_out_f)
    tgt_out_f = os.path.join(out_dir, f"{pref}_{src}_{tgt}.{tgt}")
    write_file(tgt_data, tgt_out_f)

def remove_data(og_data, exclude_data):
    new_data = []
    for src_word, tgt_word in og_data:
        for ex_src_word, ex_tgt_word in exclude_data:
            if (src_word in ex_src_word) or (tgt_word in ex_tgt_word):
                pass
            else:
                new_data.append((src_word, tgt_word))

def make_unique(data):
    items = set()
    new_data = []
    for item in data:
        if item not in items:
            new_data.append(item)
        items.add(item)
    return new_data

def read_data(fs):
    data = []
    for f in fs:
        with open(f) as inf:
            lines = [line.strip() for line in inf.readlines()]
            data += lines
    return data

def write_file(data, f):
    print("writing", f)
    with open(f, "w") as outf:
        outf.write("\n".join(data) + "\n")

def reverse_data(data):
    reverse = []
    for src, tgt, dist in data:
        reverse.append((tgt, src, dist))
    return reverse

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data", help="fastalign formatted word-list")
    # parser.add_argument("--src", help="name of src lang in data, assuming format `src ||| tgt ||| distance`, or if passing --REVERSE, `tgt ||| src ||| distance`")
    # parser.add_argument("--tgt", help="name of tgt lang in data, assuming format `src ||| tgt ||| distance`, or if passing --REVERSE, `tgt ||| src ||| distance`")
    # parser.add_argument("--REVERSE", action="store_true", help="pass this if you want to swap src and tgt")
    parser.add_argument("--src_data", help="comma-delimited list, must correspond with tgt_data")
    parser.add_argument("--tgt_data", help="comma-delimited list, must correspond with src_data")
    parser.add_argument("--src", help="name of src lang")
    parser.add_argument("--tgt", help="name of tgt lang")
    parser.add_argument("--out_dir")
    parser.add_argument("--prefix", choices=["test", "train", "fine_tune"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--EXCLUDE_SRC", default="", help="comma-delimited, must be in same order as / correspond to EXCLUDE_TGT")
    parser.add_argument("--EXCLUDE_TGT", default="", help="comma-delimited, must be in same order as / correspond to EXCLUDE_SRC")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("##################")
    print("# format_data.py #")
    print("##################")
    args = get_args()
    src_data_fs = [t.strip() for t in args.src_data.split(",")]
    tgt_data_fs = [t.strip() for t in args.tgt_data.split(",")]

    EXCLUDE_SRC = [e_item.strip() for e_item in args.EXCLUDE_SRC.split(",")]
    EXCLUDE_TGT = [e_item.strip() for e_item in args.EXCLUDE_TGT.split(",")]
    assert len(EXCLUDE_SRC) == len(EXCLUDE_TGT)

    assert len(src_data_fs) == len(tgt_data_fs)
    format(
        src_data_fs=src_data_fs,
        tgt_data_fs=tgt_data_fs,
        src=args.src,
        tgt=args.tgt,
        out_dir=args.out_dir,
        pref=args.prefix,
        seed=args.seed,
        EXCLUDE_SRC=EXCLUDE_SRC,
        EXCLUDE_TGT=EXCLUDE_TGT
    )
