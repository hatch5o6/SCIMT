import argparse
import json
import os
import csv
from datetime import datetime

FORMAT_DIV = {"train": "train", "val": "fine_tune", "test": "test"}

def log_data(
    formatted_data_dir,
    lang_pair,
    LOG_F
):
    assert LOG_F.endswith(".json")
    divs = ["train", "val", "test"]
    if os.path.exists(LOG_F):
        with open(LOG_F) as inf:
            log = json.load(inf)
        assert "history" in log.keys()
        assert "latest" in log.keys()
    else:
        log = {"latest": {}, "history": {}}
    
    cur_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    for div in divs:
        lang1, lang2 = tuple(lang_pair.split("-"))
        lang1 = lang1.strip()
        lang2 = lang2.strip()
        data_f1 = os.path.join(formatted_data_dir, f"{FORMAT_DIV[div]}_{lang1}_{lang2}.{lang1}")
        data_f2 = os.path.join(formatted_data_dir, f"{FORMAT_DIV[div]}_{lang1}_{lang2}.{lang2}")
        with open(data_f1) as inf:
            data1 = [l.strip() for l in inf.readlines()]
        with open(data_f2) as inf:
            data2 = [l.strip() for l in inf.readlines()]
        assert len(data1) == len(data2)
        
        lang_pair = f"{lang1}-{lang2}"
        print("logging lang_pair")
        if lang_pair not in log["history"]:
            log["history"][lang_pair] = {}
        if cur_datetime not in log["history"][lang_pair]:
            log["history"][lang_pair][cur_datetime] = {}

        if lang_pair not in log["latest"]:
            log["latest"][lang_pair] = {}
        log["latest"][lang_pair]["date"] = cur_datetime
        if div not in log["latest"][lang_pair]:
            log["latest"][lang_pair][div] = {}

        log["history"][lang_pair][cur_datetime][div] = len(data1)
        log["latest"][lang_pair][div] = len(data1)

        
    with open(LOG_F, "w") as outf:
        outf.write(json.dumps(log, ensure_ascii=False, indent=2))
    
    csv_f = LOG_F[:-4] + "csv"
    write_csv(log, csv_f)

def write_csv(log, csv_f):
    header = ["lang_pair", "train", "val", "test", "date_of_count"]
    with open(csv_f, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)
        for lang_pair, div_data in log["latest"].items():
            row = [lang_pair, div_data["train"], div_data["val"], div_data["test"], div_data["date"]]
            writer.writerow(row)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--formatted_data_dir")
    parser.add_argument("-l", "--lang_pair")
    parser.add_argument("-L", "--LOG_F", default="cognate_dataset_log.json")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("##########################")
    print("# cognate_dataset_log.py #")
    print("##########################")
    args = get_args()
    log_data(
        formatted_data_dir=args.formatted_data_dir,
        lang_pair=args.lang_pair,
        LOG_F=args.LOG_F
    )
