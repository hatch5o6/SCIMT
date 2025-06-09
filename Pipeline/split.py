import argparse
import random
import os
import shutil

def split_data(
    data1_f,
    data2_f,
    train_rat,
    val_rat,
    test_rat,
    seed,
    out_dir,
    UNIQUE_TEST
):
    assert train_rat + val_rat + test_rat == 1

    random.seed(seed)

    if not os.path.exists(out_dir):
        print("creating", out_dir)
        os.mkdir(out_dir)

    train1_out_f = get_file_name(data1_f, f"train-s={seed}", out_dir)
    val1_out_f = get_file_name(data1_f, f"val-s={seed}", out_dir)
    test1_out_f = get_file_name(data1_f, f"test-s={seed}", out_dir)

    train2_out_f = get_file_name(data2_f, f"train-s={seed}", out_dir)
    val2_out_f = get_file_name(data2_f, f"val-s={seed}", out_dir)
    test2_out_f = get_file_name(data2_f, f"test-s={seed}", out_dir)

    with open(data1_f) as inf:
        data1 = [
            line.strip() for line in inf
        ]
    with open(data2_f) as inf:
        data2 = [
            line.strip() for line in inf
        ]
    assert len(data1) == len(data2)
    data = list(zip(data1, data2))

    random.shuffle(data)

    train_end = round(train_rat * len(data))
    val_end = train_end + round(val_rat * len(data))

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    print("asserting split occurred correctly")
    assert train + val + test == data
    print("\tit passed :)")

    write_split(train, train1_out_f, train2_out_f)
    write_split(val, val1_out_f, val2_out_f)

    if UNIQUE_TEST:
        print("MAKING TEST SOURCE-SIDE UNIQUE")
        unique_test = {}
        for src, tgt in test:
            if src in unique_test:
                continue
            else:
                unique_test[src] = tgt
        test = [(src, tgt) for src, tgt in unique_test.items()]
    else:
        print("NORMAL TEST")
    
    write_split(test, test1_out_f, test2_out_f)

def write_split(data, f1, f2):
    with open(f1, "w") as outf1, open(f2, "w") as outf2:
        for seq1, seq2 in data:
            outf1.write(seq1.strip() + "\n")
            outf2.write(seq2.strip() + "\n")

def get_file_name(og_f, insert, new_dir):
    EXT = og_f.split(".")[-1]
    new_f = og_f[:-len(EXT)] + f"{insert}.{EXT}"
    assert new_f != og_f

    new_f_name = new_f.split("/")[-1]
    new_f = os.path.join(new_dir, new_f_name)
    assert new_f != og_f
    print("Creating file", new_f)
    return new_f

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1")
    parser.add_argument("--data2")
    parser.add_argument("--train", type=float)
    parser.add_argument("--val", type=float)
    parser.add_argument("--test", type=float)
    parser.add_argument("--seed", type=int, default=1420)
    parser.add_argument("--out_dir")
    parser.add_argument("--UNIQUE_TEST", action="store_true")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t- {k}: {v}")
    print("----------------------\n")
    return args

if __name__ == "__main__":
    print("############")
    print("# split.py #")
    print("############")
    args = get_args()
    split_data(
        args.data1,
        args.data2,
        args.train,
        args.val,
        args.test,
        args.seed,
        args.out_dir,
        args.UNIQUE_TEST
    )