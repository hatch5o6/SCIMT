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
    cap_val_size,
    UNIQUE_TEST,
    min_val_size=250,
    max_val_size=1100
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

    assert test_rat == 0
    assert val_rat + train_rat == 1
    val_amount = round(val_rat * len(data))
    print(f"UNBOUNDED VAL AMMOUNT: {val_amount}")
    # bound val_amount between 200 and 1000
    val_amount = max(min_val_size, val_amount)
    val_amount = min(max_val_size, val_amount)
    print(f"VAL AMMOUNT BOUNDED ({min_val_size}-{max_val_size}): {val_amount}")
    train_amount = len(data) - val_amount

    train_end = train_amount
    val_end = train_end + val_amount

    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    assert test == []

    print("asserting split occurred correctly")
    assert train + val + test == data
    print("\tit passed :)")

    if UNIQUE_TEST:
        print("MAKING TEST AND VAL SOURCE-SIDE UNIQUE")
        print(f"\tTEST before: {len(test)}")
        unique_test = {}
        for src, tgt in test:
            if src in unique_test:
                continue
            else:
                unique_test[src] = tgt
        test = [(src, tgt) for src, tgt in unique_test.items()]
        print(f"\tTEST after: {len(test)}")

        print(f"\n\tVAL before: {len(val)}")
        unique_val = {}
        for src, tgt in val:
            if src in unique_val:
                continue
            else:
                unique_val[src] = tgt
        val = [(src, tgt) for src, tgt in unique_val.items()]
        print(f"\tVAL after: {len(val)}")
    else:
        print("NORMAL TEST AND VAL")
    
    assert sorted(test) == sorted(list(set(test))), f"TEST COGNATES FAILED TO BE SOURCE-SIDE UNIQUE"
    assert sorted(val) == sorted(list(set(val))), f"VAL COGNATES FAILED TO BE SOURCE-SIDE UNIQUE"

    if cap_val_size:
        print(f"CAP_VAL_SIZE was set: {cap_val_size}")
        print(f"val size was {len(val)}")
        val = val[:cap_val_size]
        print(f"val size now {len(val)}")

    write_split(train, train1_out_f, train2_out_f)
    write_split(val, val1_out_f, val2_out_f)
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
    parser.add_argument("--cap_val_size")
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
    if args.cap_val_size == "null":
        cap_val_size = None
    else:
        cap_val_size = int(args.cap_val_size)
    split_data(
        args.data1,
        args.data2,
        args.train,
        args.val,
        args.test,
        args.seed,
        args.out_dir,
        cap_val_size,
        args.UNIQUE_TEST
    )