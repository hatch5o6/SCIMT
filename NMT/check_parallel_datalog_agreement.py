import argparse
import csv
import json


def main(
    bash_version_f,
    pipeline_version_f,
    NMT_version_f
):
    bash_data = read_bash_version(bash_version_f)
    pipeline_data = read_pipeline_version(pipeline_version_f)
    NMT_data = read_NMT_version(NMT_version_f)

    for lp in ["en-en", "hi-hi"]:
        assert lp in pipeline_data
        assert lp in NMT_data
        assert lp not in bash_data

    assert sorted(list(bash_data.keys()) + ["en-en", "hi-hi"]) == sorted(list(pipeline_data.keys()))

    for lang_pair in NMT_data:
        assert lang_pair in pipeline_data
        assert NMT_data[lang_pair] == pipeline_data[lang_pair]

    for lang_pair in bash_data:
        assert bash_data[lang_pair] == pipeline_data[lang_pair]

    print("Passed all tests!")

    print("pairs in pipeline that are not in bash")
    for lp in pipeline_data:
        if lp not in bash_data:
            print("\t", lp)
    print("pairs in pipeline that are not in NMT")
    for lp in pipeline_data:
        if lp not in NMT_data:
            print("\t", lp)
    

def read_bash_version(f):
    data = {}
    with open(f) as inf:
        line = inf.readline()
        while line:
            line = line.rstrip()
            assert is_data_dir_line(line)
            while is_data_dir_line(line):
                line = line.rstrip()
                assert line.endswith("/CharLOTTE_arabic") or line.endswith("/CharLOTTE_data")
                line = inf.readline().rstrip()
                assert is_lang_pair_line(line)
                while is_lang_pair_line(line):
                    lang_pair = line.split("/")[-1].strip()
                    cts = {}
                    line = inf.readline()
                    assert is_count_line(line)
                    while is_count_line(line):
                        line = line.rstrip()
                        set_ct = int(line.strip().split(":")[0].strip())
                        if is_train_line(line):
                            if "train" not in cts:
                                cts["train"] = set_ct
                            else:
                                assert cts["train"] == set_ct, f"train conflict: {set_ct} >> {cts['train']} `{line}`"
                        elif is_val_line(line):
                            if "val" not in cts:
                                cts["val"] = set_ct
                            else:
                                assert cts["val"] == set_ct, f"val conflict: {set_ct} >> {cts['val']} `{line}`"
                        elif is_test_line(line):
                            if "test" not in cts:
                                cts["test"] = set_ct
                            else:
                                assert cts["test"] == set_ct, f"test conflict: {set_ct} >> {cts['test']} `{line}`"
                        else:
                            assert line.strip().endswith(".notes") or line.strip().endswith(".500.txt")
                        line = inf.readline()
                    assert lang_pair not in data
                    data[lang_pair] = cts
                
            line = inf.readline()
    return data

def is_data_dir_line(line):
    return line.startswith("Lookin in /home")
def is_lang_pair_line(line):
    return line.startswith("    /home")
def is_count_line(line):
    return line.startswith("        ")
def is_train_line(line):
    return line.split("/")[-1].startswith("train.") and not line.strip().endswith(".notes")
def is_val_line(line):
    return line.split("/")[-1].startswith("val.") and not line.strip().endswith(".notes")
def is_test_line(line):
    return line.split("/")[-1].startswith("test.") and not line.strip().endswith(".notes")

def read_pipeline_version(f):
    rows = read_csv(f)
    header = rows[0]
    assert header == ["cfg","lang_pair","train","val","test","date_of_count"]
    data = rows[1:]
    counts = {}
    for cfg, lang_pair, train, val, test, date in data:
        assert lang_pair not in counts
        counts[lang_pair] = {"train": int(train), "val": int(val), "test": int(test)}
    return counts

def is_skip_row(row):
    skip_conditions = ["CHAR", "AUGMENT", ".TEST", ".no_grad_clip"]
    for condition in skip_conditions:
        if condition in row[0]:
            return True
    return False

def is_empty_row(row):
    for cell in row:
        if cell.strip() != "":
            return False
    return True

def is_sc_tl2cl_tl_pretrain_row(row):
    model = row[0]
    src = row[1]
    tgt = row[2]
    return model == f"{src}-{tgt}/PRETRAIN.SC_{tgt}2{src}-{tgt}.ATT.yaml"

def read_NMT_version(f):
    counts = {}
    rows = read_csv(f)
    headings = rows[0]
    headings = {head: i for i, head in enumerate(headings) if head.strip() != ""}
    # print(headings)
    data = rows[1:]
    for r, row in enumerate(data):
        if is_empty_row(row): continue
        if is_skip_row(row): continue

        model_idx = headings["config"]
        train_idx = headings["train"]
        val_idx = headings["val"]
        test_idx = headings["test"]
        src_idx = headings["src"]
        tgt_idx = headings["tgt"]

        train = int(row[train_idx].replace(",", ""))
        val = int(row[val_idx].replace(",", ""))
        test = int(row[test_idx].replace(",", ""))
    
        src = row[src_idx]
        tgt = row[tgt_idx]
        pair = f"{src}-{tgt}"

        if is_sc_tl2cl_tl_pretrain_row(row):
            pair = f"{tgt}-{tgt}"

        pair_counts = {"train": train, "val": val, "test": test}
        
        if pair in counts:
            assert counts[pair] == pair_counts, f"counts disagreement: {r}) {row[model_idx]} {pair_counts} >> {counts[pair]}"
        else:
            counts[pair] = pair_counts
    return counts


def read_csv(f):
    with open(f) as inf:
        lines = [l for l in csv.reader(inf)]
    return lines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bash_version", default="/home/hatch5o6/Cognate/code/Pipeline/log_parallel_dataset_sizes_bash.sh.out")
    parser.add_argument("-p", "--pipeline_version", default="/home/hatch5o6/Cognate/code/NMT_parallel_dataset_log.csv")
    parser.add_argument("-n", "--NMT_version", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/data_params_log.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        args.bash_version,
        args.pipeline_version,
        args.NMT_version
    )