import os
import csv


def main():
    path = "/home/hatch5o6/Cognate/code/NMT/data/PLAIN"
    n_dirs = 0
    for d in os.listdir(path):
        d_path = os.path.join(path, d)
        assert os.path.isdir(d_path)
        if not d.endswith("_dev_test"):
            continue
        n_dirs += 1
        if d == "en-fon_dev_test":
            print("--------------------------------------------------------")
            print(d_path)
            print("Asserting en-fon_dev_test is empty")
            assert os.listdir(d_path) == []
            continue

        print("--------------------------------------------------------")
        print(d_path)

        fs = sorted(os.listdir(d_path))
        assert "test.csv" in fs
        assert "val.csv" in fs
        for f in fs:
            print(f"\t-`{f}`")

        if has_old(fs):
            assert sorted(fs) == ["OLD_test.csv", "OLD_val.csv", "test.csv", "val.csv"]
            assert len(fs) == 4

            old_test_f = os.path.join(d_path, "OLD_test.csv")
            old_val_f = os.path.join(d_path, "OLD_val.csv")
            test_f = os.path.join(d_path, "test.csv")
            val_f = os.path.join(d_path, "val.csv")

            old_test_src, old_test_tgt, old_test_src_path, old_test_tgt_path = read_csv(old_test_f, type="OLD_TEST")
            old_val_src, old_val_tgt, old_val_src_path, old_val_tgt_path = read_csv(old_val_f, type="OLD_VAL")

            test_src, test_tgt, test_src_path, test_tgt_path = read_csv(test_f, type="TEST")
            val_src, val_tgt, val_src_path, val_tgt_path = read_csv(val_f, type="VAL")

            print("assert src lang is same across all files:", old_test_src)
            assert old_test_src == old_val_src == test_src == val_src
            print("assert tgt lang is same across all files:", old_test_tgt)
            assert old_test_tgt == old_val_tgt == test_tgt == val_tgt

            print("\n\nCOMPARING OLD TEST SRC to TEST SRC")
            test_src_diffs = compare_files(old_test_src_path, test_src_path)
            print("\n\nCOMPARING OLD TEST TGT to TEST TGT")
            test_tgt_diffs = compare_files(old_test_tgt_path, test_tgt_path)

            print("\n\nCOMPARING OLD VAL SRC to VAL SRC")
            val_src_diffs = compare_files(old_val_src_path, val_src_path)
            print("\n\nCOMPARING OLD VAL TGT to VAL TGT")
            val_tgt_diffs = compare_files(old_val_tgt_path, val_tgt_path)

        else:
            for f in fs:
                print("ensure f not flores200 and not FLORES+:", f)
                fpath = os.path.join(d_path, f)
                fsrc, ftgt, fsrc_path, ftgt_path = read_csv(fpath, type="NONE")
                print("\tfsrc:", fsrc)
                print("\tftgt:", ftgt)
                print("\tfsrc_path:", fsrc_path)
                print("\tftgt_path:", ftgt_path)
                assert "flores200" not in fsrc_path
                assert "FLORES+" not in fsrc_path
                assert "flores200" not in ftgt_path
                assert "FLORES+" not in ftgt_path
                print("\npassed :)\n\n")

    print(f"\n\nCHECKED {n_dirs} _dev_test dirs!")


def compare_files(f1, f2):
    diffs = []
    f1_lines = read_f(f1)
    f2_lines = read_f(f2)
    assert len(f1_lines) == len(f2_lines)
    pairs = list(zip(f1_lines, f2_lines))
    for i, (f1line, f2line) in enumerate(pairs):
        if f1line != f2line:
            diffs.append((i, f1line, f2line))
    
    print(f"Comparing:")
    print(f"\tF1:", f1)
    print(f"\tF2:", f2)
    print("\tN_DIFFS:", len(diffs))
    for i, f1line, f2line in diffs:
        print(f"................. ({i}) .................")
        print(f"F1: `{f1line}`")
        print(f"F2: `{f2line}`")

    return diffs

def read_csv(f, type):
    print("\nTYPE:", type)
    print(f"\t-f `{f}`")

    with open(f, newline='') as inf:
        rows = [row for row in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang","tgt_lang","src_path","tgt_path"]
    data = rows[1:]
    assert len(data) == 1
    src, tgt, src_path, tgt_path = tuple(data[0])
    print(f"\t-src `{src}`")
    print(f"\t-tgt `{tgt}`")
    print(f"\t-src_path `{src_path}`")
    print(f"\t-tgt_path `{tgt_path}`")
    return src, tgt, src_path, tgt_path

def read_f(f):
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

def has_old(fs):
    if "OLD_test.csv" in fs:
        assert "OLD_val.csv" in fs
        return True
    
    if "OLD_val.csv" in fs:
        assert "OLD_test.csv" in fs
        return True # should never actually return here. The assertion really is catch instances with OLD_val without OLD_test
    
    assert "OLD_test.csv" not in fs
    assert "OLD_val.csv" not in fs

    return False

if __name__ == "__main__":
    main()
