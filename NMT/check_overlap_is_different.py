import argparse
import csv
import os
import json
from tqdm import tqdm

def main(
    directory,
    tag
):
    print("getting language pairs")
    pairs = get_langs_from_dir(directory)
    # final_results = {}
    visited = set()
    print("running comparisons")
    for src_lang, tgt_lang in tqdm(pairs):
        assert (src_lang, tgt_lang) not in visited
        visited.add((src_lang, tgt_lang))

        print(f"###################### {src_lang}-{tgt_lang} #######################")
        train_dir = os.path.join(directory, f"{src_lang}-{tgt_lang}")
        dev_test_dir = os.path.join(directory, f"{src_lang}-{tgt_lang}_dev_test")

        train_file = os.path.join(train_dir, "train.csv")
        train_no_file = os.path.join(train_dir, f"train.{tag}.csv")
        train_results = compare_csvs(
            csv_f=train_file,
            csv_no_f=train_no_file,
            tag=tag
        )

        val_file = os.path.join(dev_test_dir, "val.csv")
        val_no_file = os.path.join(dev_test_dir, f"val.{tag}.csv")
        val_results = compare_csvs(
            csv_f=val_file,
            csv_no_f=val_no_file,
            tag=tag
        )
        print("-----------------------------------")
        print("TRAIN:")
        print("   TRAIN_FILE:", train_file)
        print("TRAIN_NO_FILE:", train_no_file)
        print(json.dumps(train_results, ensure_ascii=False, indent=2))

        print("-----------------------------------")
        print("VAL:")
        print("   VAL_FILE:", val_file)
        print("VAL_NO_FILE:", val_no_file)
        print(json.dumps(val_results, ensure_ascii=False, indent=2))

        print("\n\n\n")
        # assert (src_lang, tgt_lang) not in final_results
        # final_results[(src_lang, tgt_lang)] = {
        #     "train_results": {
        #         "og_file": train_file,
        #         "no_file": train_no_file,
        #         "results": train_results
        #     }, 
        #     "val_results": {
        #         "og_file": val_file,
        #         "no_file": val_no_file,
        #         "results": val_results
        #     }
        # }

def compare_csvs(csv_f, csv_no_f, tag):
    csv_content = read_csv_file(csv_f)
    csv_no_content = read_csv_file(csv_no_f)
    assert len(csv_content) == len(csv_no_content)

    results = []
    for i in range(len(csv_content)):
        src_lang, tgt_lang, src_path, tgt_path = csv_content[i]
        no_src_lang, no_tgt_lang, no_src_path, no_tgt_path = csv_no_content[i]
        assert src_lang == no_src_lang
        assert tgt_lang == no_tgt_lang

        assert no_src_path.replace(f".{tag}", "") == src_path
        assert no_tgt_path.replace(f".{tag}", "") == tgt_path

        src_content = read_f(src_path)
        no_src_content = read_f(no_src_path)
        tgt_content = read_f(tgt_path)
        no_tgt_content = read_f(no_tgt_path)

        pair = f'{src_lang}-{tgt_lang}'
        assert pair not in results
        results.append({
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,

            "src_path": src_path,
            "no_src_path": no_src_path,
            "tgt_path": tgt_path,
            "no_tgt_path": no_tgt_path,

            "src_match": src_content == no_src_content,
            "tgt_match": tgt_content == no_tgt_content
        })
    return results

def read_f(f):
    with open(f) as inf:
        content = inf.read()
    return content
    
def read_csv_file(csv_f):
    with open(csv_f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    data = [tuple(r) for r in rows[1:]]
    return data

def get_langs_from_dir(directory):
    pairs = set()
    for f in tqdm(os.listdir(directory)):
        f_path = os.path.join(directory, f)
        assert os.path.isdir(f_path)

        lang_pair = f.split("_")[0]
        src_lang, tgt_lang = lang_pair.split("-")
        assert src_lang.strip() != ""
        assert tgt_lang.strip() != ""
        assert f in [f"{src_lang}-{tgt_lang}", f"{src_lang}-{tgt_lang}_dev_test"]
        pairs.add((src_lang, tgt_lang))
    return sorted(list(pairs))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="/home/hatch5o6/Cognate/code/NMT/data/PLAIN")
    parser.add_argument("--tag", default="no_overlap_v1")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"--{k}=`{v}`")
    print("--------------\n\n")
    return args

if __name__ == "__main__":
    print("#################################")
    print("# check_overlap_is_different.py #")
    print("#################################")
    args = get_args()
    main(
        args.dir,
        args.tag
    )