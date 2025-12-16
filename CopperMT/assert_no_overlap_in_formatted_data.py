import argparse
import os
import json
import copy
from NMT.assert_no_data_overlap import assert_no_overlap

def test_only(
    format_out_dir,
    src,
    tgt
):
    print("FORMATTED DATA: testing no overlap ONLY:")
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
    print(json.dumps(results, ensure_ascii=False, indent=2))
    assert passed in [True, False]
    if passed == True:
        print("PASSED :) -- there is no overlap in the formatted data")
    elif passed == False:
        print("Failed. The formatted data contains overlap. Will now exit.")

    assert passed == True


def main(
    format_out_dir,
    src,
    tgt
):
    print("FORMATTED DATA: Will remove overlap and then test.")
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
    copy_test_pairs = copy.deepcopy(test_pairs)

    len_train_pairs = len(train_pairs)
    len_fine_tune_pairs = len(fine_tune_pairs)
    len_test_pairs = len(test_pairs)

    print("First removing the overlap between train / fine_tune / test")
    new_train_pairs, new_fine_tune_pairs = remove_overlap(
        train_pairs=train_pairs, 
        fine_tune_pairs=fine_tune_pairs, 
        test_pairs=test_pairs
    )
    assert test_pairs == copy_test_pairs

    print("REDUCTIONS:")
    print(f"\t-      TEST: {len_train_pairs} > {len(new_train_pairs)}")
    print(f"\t- FINE_TUNE: {len_fine_tune_pairs} > {len(new_fine_tune_pairs)}")
    assert len_test_pairs == len(test_pairs)
    print(f"\t-      TEST: {len_test_pairs} = {len(test_pairs)}")

    print("Now testing that there is no overlap")
    passed, results = assert_no_overlap(
        train=new_train_pairs,
        dev=new_fine_tune_pairs,
        test=test_pairs,
        VERBOSE=True
    )
    assert passed in [True, False]
    print("OVERLAP IN FORMATTED COGNATE DATA:")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if passed == True:
        for val in results.values():
            assert val == True
        print("\tTest passed :)")
    elif passed == False:
        print("\t!!!! Test failed !!!!")
        print("\tExiting")
    assert passed == True

    print("Now we have proven there is no overlap, writing the new train and fine_tune_files")
    write_new_set(
        pairs=new_train_pairs,
        prefix="train",
        src=src,
        tgt=tgt,
        format_out_dir=format_out_dir
    )
    write_new_set(
        pairs=new_fine_tune_pairs,
        prefix="fine_tune",
        src=src,
        tgt=tgt,
        format_out_dir=format_out_dir
    )

def write_new_set(pairs, prefix, src, tgt, format_out_dir):
    assert prefix in ["train", "fine_tune"]

    src_out_f = os.path.join(format_out_dir, f"{prefix}_{src}_{tgt}.{src}")
    tgt_out_f = os.path.join(format_out_dir, f"{prefix}_{src}_{tgt}.{tgt}")
    assert os.path.exists(src_out_f)
    assert os.path.exists(tgt_out_f)

    print(f"Overwriting {src_out_f} with overlap removed")
    print(f"Overwriting {tgt_out_f} with overlap removed")

    with open(src_out_f, "w") as sf, open(tgt_out_f, "w") as tf:
        for src_seg, tgt_seg in pairs:
            sf.write(src_seg.strip() + "\n")
            tf.write(tgt_seg.strip() + "\n")
    return src_out_f, tgt_out_f

def remove_overlap(train_pairs, fine_tune_pairs, test_pairs):
    # train_src_set, train_tgt_set =              get_src_tgt_sets(train_pairs)
    fine_tune_src_set, fine_tune_tgt_set =      get_src_tgt_sets(fine_tune_pairs)
    test_src_set, test_tgt_set =                get_src_tgt_sets(test_pairs)

    new_train_pairs = []
    for train_src_seg, train_tgt_seg in train_pairs:
        REMOVE = False
        if any([
            train_src_seg in fine_tune_src_set,
            train_src_seg in test_src_set,

            train_tgt_seg in fine_tune_tgt_set,
            train_tgt_seg in test_tgt_set
        ]):
            REMOVE = True
        if not REMOVE:
            new_train_pairs.append((train_src_seg, train_tgt_seg))
    
    new_fine_tune_pairs = []
    for fine_tune_src_seg, fine_tune_tgt_seg in fine_tune_pairs:
        REMOVE = False
        if any([
            fine_tune_src_seg in test_src_set,

            fine_tune_tgt_seg in test_tgt_set
        ]):
            REMOVE = True
        if not REMOVE:
            new_fine_tune_pairs.append((fine_tune_src_seg, fine_tune_tgt_seg))
            
    return new_train_pairs, new_fine_tune_pairs

def get_src_tgt_sets(pairs):
    src_set = set([src for src, tgt in pairs])
    tgt_set = set([tgt for src, tgt in pairs])
    return src_set, tgt_set

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
    parser.add_argument("--TEST_ONLY", action="store_true", default=False)

    args = parser.parse_args()
    print("Arguments:--------")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("\n------------------\n\n")
    return args

if __name__ == "__main__":
    print("##########################################")
    print("# assert_no_overlap_in_formatted_data.py #")
    print("##########################################")
    args = get_args()
    if args.TEST_ONLY == True:
        test_only(
            format_out_dir=args.format_out_dir,
            src=args.src,
            tgt=args.tgt
        )
    else:
        main(
            format_out_dir=args.format_out_dir,
            src=args.src,
            tgt=args.tgt
        )