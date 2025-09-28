import csv
import yaml
import os
import argparse
from datetime import datetime


def main(
    configs_dir,
    out
):
    with open(out, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["config", "train", "val", "test", "date"])
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for lang_d in os.listdir(configs_dir):
            lang_d_path = os.path.join(configs_dir, lang_d)
            print("LANG D PATH:", lang_d_path)
            for f in os.listdir(lang_d_path):
                f_path = os.path.join(lang_d_path, f)
                print(f"F: {f}")
                config = read_config(f_path)

                sc_model_id = config["sc_model_id"]
                n_train = count_lines_in_csv(config["train_data"], sc_model_id)
                n_val = count_lines_in_csv(config["val_data"], sc_model_id)
                n_test = count_lines_in_csv(config["test_data"], sc_model_id)

                config_name = f"{lang_d}/{f}"

                writer.writerow([config_name, n_train, n_val, n_test, date])

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def count_lines_in_csv(f, sc_model_id):
    with open(f, newline="") as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    data = [tuple(r) for r in rows[1:]]
    n_lines = 0
    for src, tgt, src_path, tgt_path in data:
        assert "{SC_MODEL_ID}" not in tgt_path
        if "{SC_MODEL_ID}" in src_path:
            src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)

        src_lines = count_lines_in_f(src_path)
        tgt_lines = count_lines_in_f(tgt_path)
        assert src_lines == tgt_lines
        n_lines += src_lines
    return n_lines

def count_lines_in_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return len(lines)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("--out", default="/home/hatch5o6/Cognate/code/NMT/data_log.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        configs_dir=args.configs_dir,
        out=args.out
    )