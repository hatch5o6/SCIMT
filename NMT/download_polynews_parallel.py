import argparse
import os
import shutil
from tqdm import tqdm
import json
from datasets import load_dataset

DIR = "/home/hatch5o6/nobackup/archive/data/polynews_parallel"

def get_data(
    langs
):
    if os.path.exists(DIR):
        print("Deleting", DIR)
        shutil.rmtree(DIR)
    print("Creating", DIR)
    os.mkdir(DIR)
    notes_f = os.path.join(DIR, "notes")
    with open(notes_f, "w") as outf:
        outf.write("Data from https://huggingface.co/datasets/aiana94/polynews-parallel")

    print("getting all possible pairs")
    pairs = []
    for src in langs:
        for tgt in langs:
            if src != tgt:
                pairs.append((src, tgt))
    
    print(f"Got {len(pairs)} possible lang pairs")
    for src, tgt in pairs:
        print("GETTING", src, tgt)
        doc_data = []
        sent_data = []
        try:
            doc_data = get_poly(src, tgt)
        except:
            print("Could not find POLYNEWS PARALLEL data for", f"{src}_{tgt}")
    
        data = doc_data + sent_data
        print(f"{len(data)} TOTAL SENT PAIRS")

        if len(data) > 0:
            out_dir = os.path.join(DIR, f"{src}-{tgt}")
            os.mkdir(out_dir)
            src_path = os.path.join(out_dir, f"{src}.txt")
            tgt_path = os.path.join(out_dir, f"{tgt}.txt")
            with open(src_path, "w") as sf, open(tgt_path, "w") as tf:
                for src_seg, tgt_seg in data:
                    sf.write(src_seg.strip() + "\n")
                    tf.write(tgt_seg.strip() + "\n")

def get_poly(src_lang, tgt_lang):
    src = []
    tgt = []
    config = f"{src_lang}-{tgt_lang}"
    # print("Getting config", config)
    data = load_dataset("aiana94/polynews-parallel", config)
    for item in data['train']:
        src.append(item['src'])
        tgt.append(item['tgt'])
    pairs = list(zip(src, tgt))
    print(f"{src_lang}-{tgt_lang}: {len(pairs)} POLYNEWS PARALLEL Pairs")
    return pairs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", help="comma-delimited list of langs")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"{k}: '{v}'")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("-------------------------------------------")
    print("###### download_polynews_parallel.py ######")
    print("-------------------------------------------")
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    get_data(langs=langs)
