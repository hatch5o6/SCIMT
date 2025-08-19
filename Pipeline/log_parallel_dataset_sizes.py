import argparse
import csv
import json
import os
import configparser
from datetime import datetime

import sys
sys.path.append("/home/hatch5o6/Cognate/code/NMT")
from parallel_datasets import MultilingualDataset, ParallelDataset

def log_sizes(
    LOG_F,
    cfgs,
    nmt_data_dir
):
    assert LOG_F.endswith(".json")
    if os.path.exists(LOG_F):
        with open(LOG_F) as inf:
            log = inf.read()
        if log.strip() != "":
            log = json.loads(log)
        else:
            log = {"latest":{}, "history":{}}
    else:
        log = {"latest":{}, "history":{}}
    assert "history" in log
    assert "latest" in log

    FROM_NMT = False
    if cfgs is not None:
        assert nmt_data_dir is None
        cfgs_dir = cfgs
        cfgs = list(os.listdir(cfgs))
    
    if nmt_data_dir is not None:
        FROM_NMT = True
        assert cfgs is None
        cfgs = get_nmt_data_dir(nmt_data_dir)
    
    assert isinstance(cfgs, list)
    cur_datetime = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
    for f in cfgs:
        if FROM_NMT:
            assert isinstance(f, tuple)
            config, f_path = f
            assert isinstance(config, dict)
            assert isinstance(f_path, str)
        else:
            assert isinstance(f, str)
            assert f.endswith(".cfg")
            f_path = os.path.join(cfgs_dir, f)
            config = read_config(f_path)

        if f_path not in log["latest"]:
            log["latest"][f_path] = {"date": cur_datetime, "sizes": {}}
        else:
            log["latest"][f_path]["date"] = cur_datetime
        if f_path not in log["history"]:
            log["history"][f_path] = {}
        assert cur_datetime not in log["history"][f_path]
        log["history"][f_path][cur_datetime] = {}

        for div in ["train", "val", "test"]:
            data_csv = config.get(f"PARALLEL_{div.upper()}")
            if data_csv is not None:
                if div not in log["latest"][f_path]["sizes"]:
                    log["latest"][f_path]["sizes"][div] = {}
                if div not in log["history"][f_path][cur_datetime]:
                    log["history"][f_path][cur_datetime][div] = {}

                dataset_length, lengths_by_lang_pair, raw_lengths, total_raw_length = get_size_from_csv(data_csv)
                log["latest"][f_path]["sizes"][div]["dataset_length"] = dataset_length
                log["latest"][f_path]["sizes"][div]["lengths_by_pair"] = lengths_by_lang_pair
                log["latest"][f_path]["sizes"][div]["raw_lengths"] = raw_lengths
                log["latest"][f_path]["sizes"][div]["total_raw_lengths"] = total_raw_length

                log["history"][f_path][cur_datetime][div]["dataset_length"] = dataset_length
                log["history"][f_path][cur_datetime][div]["lengths_by_pair"] = lengths_by_lang_pair
                log["history"][f_path][cur_datetime][div]["raw_lengths"] = raw_lengths
                log["history"][f_path][cur_datetime][div]["total_raw_lengths"] = total_raw_length

    with open(LOG_F, "w") as outf:
        # print(log)
        outf.write(json.dumps(log, ensure_ascii=False, indent=2))

    csv_f = LOG_F[:-4] + "csv"
    write_csv(log, csv_f)

def get_nmt_data_dir(dir):
    data_subdirs = []
    for d in os.listdir(dir):
        if d in ["_OLD", "_pipeline_testing"]: continue
        d_path = os.path.join(dir, d)
        assert os.path.isdir(d_path)

        data_subdirs.append(d)

    dirs_by_l = {}
    for subdir in data_subdirs:
        lang_pair = subdir.split("_")[0]
        if lang_pair not in dirs_by_l:
            dirs_by_l[lang_pair] = []
        dirs_by_l[lang_pair].append(subdir)
    
    lang_configs = []
    for lang_pair, subdirs in dirs_by_l.items():
        assert len(subdirs) in [1, 2]
        if len(subdirs) == 2:
            train_dir, dev_test_dir = tuple(sorted(subdirs))
            assert train_dir == lang_pair
            assert dev_test_dir == f"{lang_pair}_dev_test"
        else:
            assert subdirs[0] in [lang_pair, f"{lang_pair}_dev_test"]
            if subdirs[0] == lang_pair:
                train_dir = subdirs[0]
                dev_test_dir = None
            else:
                train_dir = None
                dev_test_dir = subdirs[0]

        lang_config = {}

        if train_dir:
            train_path = os.path.join(dir, train_dir, "train.csv")
            if os.path.exists(train_path):
                lang_config["PARALLEL_TRAIN"] = train_path
        
        if dev_test_dir:
            val_path = os.path.join(dir, dev_test_dir, "val.csv")
            if os.path.exists(val_path):
                lang_config["PARALLEL_VAL"] = val_path
            test_path = os.path.join(dir, dev_test_dir, "test.csv")
            if os.path.exists(test_path):
                lang_config["PARALLEL_TEST"] = test_path
        
        modified_path = dev_test_dir.replace("_dev_test", "(_dev_test)")
        modified_path = os.path.join(modified_path, "(train|val|test).csv")
        lang_configs.append((lang_config, modified_path))

    return lang_configs

def read_config(f):
    TERMS = ["PARALLEL_TRAIN", "PARALLEL_VAL", "PARALLEL_TEST"]
    config = {}
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    for line in lines:
        for term in TERMS:
            if line.startswith(f"{term}="):
                content = line.split(f"{term}=")[-1]
                assert term not in config
                config[term] = content
    
    for term in TERMS:
        assert term in config
    
    return config

def write_csv(log, csv_f):
    header = ["cfg", "lang_pair", "train", "val", "test", "date_of_count"]
    with open(csv_f, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(header)
        rows_by_pair = {}
        CUR_DATE=None
        for cfg_path, sizes_data in log["latest"].items():
            cur_datetime = log["latest"][cfg_path]["date"]
            if CUR_DATE is None:
                CUR_DATE = cur_datetime
            assert cur_datetime == CUR_DATE
            for div, div_data in sizes_data["sizes"].items():
                if "lengths_by_pair" not in div_data:
                    print("##### lengths_by_pair not in div data! #####")
                    print(json.dumps(log, indent=2))
                    print("PATH:", cfg_path, "sizes", div)
                if div_data["lengths_by_pair"] is None: continue
                for pair, pair_length in div_data["lengths_by_pair"].items():
                    if pair not in rows_by_pair:
                        rows_by_pair[pair] = {k: None for k in header}
                        rows_by_pair[pair]["lang_pair"] = pair
                        rows_by_pair[pair]["cfg"] = cfg_path
                        rows_by_pair[pair]["date_of_count"] = cur_datetime
                    rows_by_pair[pair][div] = pair_length
        for pair, row in rows_by_pair.items():
            linear_row = [row[k] for k in header]
            writer.writerow(linear_row)

def get_size_from_csv(data_csv):
    if data_csv == "null":
        return None, None, None, None
    else:
        print("data_csv", data_csv)
        assert data_csv.endswith(".csv")
        assert os.path.exists(data_csv)

    dataset = MultilingualDataset(
        data_csv=data_csv,
        append_src_lang_tok=False,
        append_tgt_lang_tok=False,
        append_tgt_to_src=False,
    )
    dataset_length = len(dataset)
    lengths_by_lang_pair = dataset.lengths
    raw_lengths = dataset.raw_lengths

    total_pair_length = 0
    for lang_pair, pair_length in lengths_by_lang_pair.items():
        assert isinstance(pair_length, int)
        total_pair_length += pair_length
    # print("--")
    # print("dataset_length", dataset_length)
    # print("total_pair_length", total_pair_length)
    assert dataset_length == total_pair_length

    total_raw_length = 0
    for k, k_length in raw_lengths.items():
        assert isinstance(k_length, int)
        total_raw_length += k_length
    
    return dataset_length, lengths_by_lang_pair, raw_lengths, total_raw_length


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--LOG_F", default="NMT_parallel_dataset_log.json")
    parser.add_argument("--cfgs", 
                        # default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC", 
                        help="dir holding cfgs files (for SC data)")
    parser.add_argument("--nmt_data_dir", 
                        default="/home/hatch5o6/Cognate/code/NMT/data",
                        help="NMT parallel data option"
                        )
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("#################################")
    print("# log_parallel_dataset_sizes.py #")
    print("#################################")
    args = get_args()
    log_sizes(
        LOG_F=args.LOG_F,
        nmt_data_dir=args.nmt_data_dir,
        cfgs=args.cfgs
    )