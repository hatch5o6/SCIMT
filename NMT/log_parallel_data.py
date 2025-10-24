import csv
import yaml
import os
import argparse
from datetime import datetime

from train import read_config


def main(
    configs_dir,
    out
):
    with open(out, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["config", "train", "val", "test", "date"])
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for lang_d in os.listdir(configs_dir):
            if lang_d.endswith(".csv"): continue
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

def log_data_and_training_params(
    configs_dir,
    out
):
    with open(out, "w", newline="") as outf:
        writer = csv.writer(outf)
        header = [
            "config",
            "sc_model_id",
            "upsample",
            "device", 
            "n_gpus", 
            "seed", 
            "qos", 
            "early_stop",
            "save_top_k", 
            "max_steps", 
            "warmup_steps",
            "batch_size (train/val/test)",
            "val_interval", 
            "learning_rate", 
            "weight_decay",
            "train", 
            "val", 
            "test",
            "date"
        ]
        writer.writerow(header)
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for lang_d in os.listdir(configs_dir):
            if lang_d.endswith(".csv"): continue
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

                row = []
                for key in header:
                    if key == "config":
                        row.append(config_name)
                    elif key == "batch_size (train/val/test)":
                        row.append(f"{config['train_batch_size']}/{config['val_batch_size']}/{config['test_batch_size']}")
                    elif key == "train":
                        row.append(f"{n_train:,}")
                    elif key == "val":
                        row.append(f"{n_val:,}")
                    elif key == "test":
                        row.append(f"{n_test:,}")
                    elif key == "date":
                        row.append(date)
                    else:
                        row.append(config[key])
                writer.writerow(row)
            writer.writerow([])

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
    parser.add_argument("--out", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/data_params_log.csv")
    parser.add_argument("--mode", default="all", choices=["sizes_only", "all"])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.mode == "sizes_only":
        main(
            configs_dir=args.configs_dir,
            out=args.out
        )
    else:
        log_data_and_training_params(
            configs_dir=args.configs_dir,
            out=args.out
        )