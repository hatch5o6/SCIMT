import argparse
import os
import json
from NMT.assert_no_data_overlap import assert_no_overlap

def main(
    format_out_dir,
    src,
    tgt
):
    format_out_dir_fs = sorted(os.listdir(format_out_dir))
    files = sorted([
        f"train_{src}_{tgt}.{src}",
        f"train_{src}_{tgt}.{tgt}",

        f"fine_tune_{src}_{tgt}.{src}",
        f"fine_tune_{src}_{tgt}.{tgt}",

        f"test_{src}_{tgt}.{src}",
        f"test_{src}_{tgt}.{tgt}",
    ])
    assert format_out_dir_fs == files

    train_pairs         = get_pairs(src, tgt, "train", format_out_dir)
    fine_tune_pairs     = get_pairs(src, tgt, "fine_tune", format_out_dir)
    test_pairs          = get_pairs(src, tgt, "test", format_out_dir)

    passed, results = assert_no_overlap(
        train=train_pairs,
        dev=fine_tune_pairs,
        test=test_pairs,
        VERBOSE=True
    )
    assert passed in [True, False]
    print("OVERLAP IN FORMATTED COGNATE DATA:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if passed == True:
        for val in results.values():
            assert val == True
        print("\tTest passed!")
    elif passed == False:
        print("\tTest failed :(")
        print("\tExiting")
    assert passed == True

def get_pairs(src, tgt, prefix, format_out_dir):
    src_file = os.path.join(format_out_dir, f"{prefix}_{src}_{tgt}.{src}")
    tgt_file = os.path.join(format_out_dir, f"{prefix}_{src}_{tgt}.{tgt}")
    src_lines = read_f(src_file)
    tgt_lines = read_f(tgt_file)
    assert len(src_lines) == len(tgt_lines)
    pairs = list(zip(src_lines, tgt_lines))
    return pairs
    
def read_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format_out_dir", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)

    args = parser.parse_args()
    print("Arguments:--------")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("\n------------------\n\n")
    return args

if __name__ == "__main__":
    print("########################")
    print("# assert_no_overlap.py #")
    print("########################")
    args = get_args()
    main(
        format_out_dir=args.format_out_dir,
        src=args.src,
        tgt=args.tgt
    )