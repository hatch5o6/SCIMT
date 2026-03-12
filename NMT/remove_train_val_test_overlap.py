import argparse
import json
import os
from assert_no_data_overlap import assert_no_overlap
from datetime import datetime

def main(
    src,
    tgt,
    train_fs,
    val_fs,
    test_fs,
    out_dir
):
    train, train_src_f, train_tgt_f = read_src_tgt_data(train_fs)
    val, val_src_f, val_tgt_f = read_src_tgt_data(val_fs)
    test, test_src_f, test_tgt_f = read_src_tgt_data(test_fs)

    passed, results = assert_no_overlap(train, val, test, VERBOSE=False)
    print("PASSED NO OVERLAP TEST?", passed)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if not passed:
        new_train, new_val, new_test = remove_overlap(train, val, test)
    else:
        new_train, new_val, new_test = train, val, test

    assert test == new_test

    print("CONFIRMING NO OVERLAP")
    passed, results = assert_no_overlap(new_train, new_val, new_test, VERBOSE=False)
    assert passed == True
    print("Confirmed. There is no overlap.")

    write_data(new_train, src, tgt, out_dir, train_src_f, train_tgt_f, div="train")
    write_data(new_val, src, tgt, out_dir, val_src_f, val_tgt_f, div="val")
    write_data(new_test, src, tgt, out_dir, test_src_f, test_tgt_f, div="test")


def write_data(pairs, src, tgt, out_dir, src_data_f, tgt_data_f, div="train"):
    assert div in ["train", "val", "test"]
    src_segs = [s.strip() for s, t in pairs]
    tgt_segs = [t.strip() for s, t in pairs]

    src_f = os.path.join(out_dir, f"{div}.{src}.txt")
    tgt_f = os.path.join(out_dir, f"{div}.{tgt}.txt")
    notes_f = os.path.join(out_dir, f"{div}.notes")

    write_file(src_segs, src_f)
    write_file(tgt_segs, tgt_f)
    write_notes(src_data_f, tgt_data_f, notes_f)

def write_notes(src_f, tgt_f, notes_f):
    with open(notes_f, "w") as outf:
        outf.write("PAIRS CAME FROM THE FOLLOWING FILES:\n")
        outf.write(f"\t- (`{src_f}`, `{tgt_f}`)\n")
        outf.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def write_file(lines, f):
    with open(f, "w") as outf:
        outf.write("\n".join(lines) + "\n")

def read_src_tgt_data(fs):
    src_f, tgt_f = get_src_tgt_paths(fs)
    src_lines = read_f(src_f)
    tgt_lines = read_f(tgt_f)
    assert len(src_lines) == len(tgt_lines)
    return list(zip(src_lines, tgt_lines)), src_f, tgt_f

def read_f(f):
    with open(f) as inf:
        return [l.strip() for l in inf.readlines()]

def get_src_tgt_paths(fs):
    return [item.strip() for item in fs.split(",")]

def remove_overlap(train, val, test):
    print("REMOVING OVERLAP")
    print("\tTRAIN:", len(train))
    print("\t  VAL:", len(val))
    print("\t TEST:", len(test))

    val_src_set, val_tgt_set = get_src_tgt_sets(val)
    test_src_set, test_tgt_set = get_src_tgt_sets(test)

    # For every src or tgt sent in train that is also in val or test, remove it
    new_train_pairs = []
    for train_src_seg, train_tgt_seg in train:
        REMOVE = any([
            train_src_seg in val_src_set,
            train_src_seg in test_src_set,

            train_tgt_seg in val_tgt_set,
            train_tgt_seg in test_tgt_set
        ])
        if not REMOVE:
            new_train_pairs.append((train_src_seg, train_tgt_seg))
    
    # For every src or tgt sent in val that is also in test, remove it
    new_val_pairs = []
    for val_src_seg, val_tgt_seg in val:
        REMOVE = any([
            val_src_seg in test_src_set,

            val_tgt_seg in test_tgt_set
        ])
        if not REMOVE:
            new_val_pairs.append((val_src_seg, val_tgt_seg))
    
    print("\t--->")
    print("\tTRAIN:", len(new_train_pairs))
    print("\t  VAL:", len(new_val_pairs))
    print("\t TEST:", len(test))
    # There should now be no overlap at all between these sets being returned, not at the pair or segment level
    return new_train_pairs, new_val_pairs, test

def get_src_tgt_sets(pairs):
    src_set = set([src for src, tgt in pairs])
    tgt_set = set([tgt for src, tgt in pairs])
    return src_set, tgt_set

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="src language")
    parser.add_argument("--tgt", help="tgt language")
    parser.add_argument("--train", help="comma-delimited tuple '<src path>,<tgt path>'")
    parser.add_argument("--val", help="comma-delimited tuple '<src path>,<tgt path>')")
    parser.add_argument("--test", help="comma-delimited tuple '<src path>,<tgt path>'")
    parser.add_argument("--out_dir", default="/home/hatch5o6/nobackup/archive/data/CharLOTTE_occitan/fr-oc")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = get_args()
    main(
        src=args.src, 
        tgt=args.tgt, 
        train_fs=args.train, 
        val_fs=args.val, 
        test_fs=args.test, 
        out_dir=args.out_dir)