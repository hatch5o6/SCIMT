import argparse
import os
import csv
import json
from tqdm import tqdm

VERBOSE_ENDINGS = ["RNN-278_S-0", "RNN-106_S-0", "RNN-42_S-0", "SMT-null_S-0"]

def main(
    CopperMT_dir,
    log_f
):
    log = read_log(log_f)
    for lang_pair, values in log.items():
        print([lang_pair] + values)
    print("\n")
    
    visited = set()
    for d in tqdm(sorted(os.listdir(CopperMT_dir))):
        verbose = False
        for ending in VERBOSE_ENDINGS:
            if d.endswith(ending):
                verbose = True
                break
        
        d_path = os.path.join(CopperMT_dir, d)
        split_d = d.split("-")
        src, tgt = split_d[:2]
        if tgt == "DJK.ATT":
            tgt = "DJK"
        src = src.lower()
        tgt = tgt.lower()

        lang_pair = f"{src}-{tgt}"

        visited.add(lang_pair)
        assert lang_pair in log, f"Lang pair {lang_pair} is not in log: {log_f}"
        parallel_data_dir = os.path.join(d_path, f"inputs/split_data/{src}_{tgt}/0")

        log_train_len, log_val_len, log_test_len, date_of_count = log[lang_pair]

        test_data = read_parallel_data(parallel_data_dir, src, tgt, div="test")
        train_data = read_parallel_data(parallel_data_dir, src, tgt, div="train")
        val_data = read_parallel_data(parallel_data_dir, src, tgt, div="fine_tune")

        if verbose:
            print(f"############ {d} #############")
            print(lang_pair)
            print("-------")
            print("log_train_len:", log_train_len)
            print("train_data:", len(train_data), "\n")
            print("log_val_len:", log_val_len)
            print("val_data:", len(val_data), "\n")
            print("log_test_len:", log_test_len)
            print("test_data:", len(test_data), "\n\n\n")

        assert len(test_data) == log_test_len, \
            f"Test data in {parallel_data_dir} does not match log: (file:{len(test_data)} != log:{log_test_len})"
        assert len(train_data) == log_train_len, \
            f"Test data in {parallel_data_dir} does not match log: (file:{len(train_data)} != log:{log_train_len})"
        assert len(val_data) == log_val_len, \
            f"Test data in {parallel_data_dir} does not match log: (file:{len(val_data)} != log:{log_val_len})"
        
    print("ALL PASSED :)")
    print("pairs visited:", visited)
    print("not visited:", set(log.keys()).difference(visited))


def read_parallel_data(directory, src, tgt, div):
    assert div in ["fine_tune", "test", "train"]
    src_f = os.path.join(directory, f"{div}_{src}_{tgt}.{src}")
    tgt_f = os.path.join(directory, f"{div}_{src}_{tgt}.{tgt}")
    src_lines = read_file(src_f)
    tgt_lines = read_file(tgt_f)
    assert len(src_lines) == len(tgt_lines)

    return list(zip(src_lines, tgt_lines))

def read_file(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    return lines

def read_log(f):
    data_dict = {}
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
        assert rows[0] == ["lang_pair", "train", "val", "test", "date_of_count"]
        rows = rows[1:]
        for lang_pair, train, val, test, data_of_count in rows:
            assert lang_pair not in data_dict
            data_dict[lang_pair] = [int(train), int(val), int(test), data_of_count]
    return data_dict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="/home/hatch5o6/Cognate/code/cognate_dataset_log_NG=True.csv")
    parser.add_argument("--CopperMT_dir", default="/home/hatch5o6/nobackup/archive/CopperMT")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(CopperMT_dir=args.CopperMT_dir, log_f=args.log)
