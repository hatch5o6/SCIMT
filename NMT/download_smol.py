import argparse
import os
import shutil
from tqdm import tqdm
import json
from datasets import load_dataset

DIR = "/home/hatch5o6/nobackup/archive/data/smol_gatitos"

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
        outf.write("Data from https://huggingface.co/datasets/google/smol")

    print("getting all possible pairs")
    pairs = []
    for src in langs:
        for tgt in langs:
            if src != tgt:
                pairs.append((src, tgt))
    
    print(f"Got {len(pairs)} possible pairs")
    for src, tgt in pairs:
        doc_data = []
        sent_data = []
        gatitos_data = []
        try:
            doc_data = get_smoldoc(src, tgt)
        except:
            print("Could not find SMOLDOC data for", f"{src}_{tgt}")
        else:
            write_data(doc_data, "SMOLDOC", src, tgt)

        try:
            sent_data = get_smolsent(src, tgt)
        except:
            print("Could not find SMOLSENT data for", f"{src}_{tgt}")
        else:
            write_data(sent_data, "SMOLSENT", src, tgt)

        try:
            gatitos_data = get_gatitos(src, tgt)
        except:
            print("Could not find GATITOS data for", f"{src}_{tgt}")
        else:
            write_data(gatitos_data, "GATITOS", src, tgt)
    
def write_data(data, dir_name, src, tgt):
    if len(data) > 0:
        print(f"WRITING {len(data)} {src}-{tgt} PAIRS IN {dir_name} DATA")
        sub_dir = os.path.join(DIR, dir_name)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        out_dir = os.path.join(sub_dir, f"{src}-{tgt}")
        print(f"\tout_dir: {out_dir}")
        assert not os.path.exists(out_dir)
        os.mkdir(out_dir)
        src_path = os.path.join(out_dir, f"{src}.txt")
        tgt_path = os.path.join(out_dir, f"{tgt}.txt")
        print("\tsrc_path:", src_path)
        print("\ttgt_path:", tgt_path)
        with open(src_path, "w") as sf, open(tgt_path, "w") as tf:
            for src_seg, tgt_seg in data:
                # print("\t\twriting", src_seg)
                sf.write(src_seg.strip() + "\n")
                tf.write(tgt_seg.strip() + "\n")

def get_smoldoc(src_lang, tgt_lang):
    src = []
    tgt = []
    config = f"smoldoc__{src_lang}_{tgt_lang}"
    # print("Getting config", config)
    data = load_dataset("google/smol", config)
    for item in data['train']:
        item_srcs = item['srcs']
        item_tgts = item['trgs']
        assert len(item_srcs) == len(item_tgts)
        src += item_srcs
        tgt += item_tgts
    pairs = list(zip(src, tgt))
    print(f"{src_lang}-{tgt_lang}: {len(pairs)} SMOLDOC Pairs")
    return pairs

def get_smolsent(src_lang, tgt_lang):
    src = []
    tgt = []
    config = f"smolsent__{src_lang}_{tgt_lang}"
    # print("getting config", config)
    data = load_dataset("google/smol", config)
    for item in data['train']:
        item_src = item['src']
        item_tgt = item['trg']
        src.append(item_src)
        tgt.append(item_tgt)
    pairs = list(zip(src, tgt))
    print(f"{src_lang}-{tgt_lang}: {len(pairs)} SMOLSENT Pairs")
    return pairs

def get_gatitos(src_lang, tgt_lang):
    src = []
    tgt = []
    config = f"gatitos__{src_lang}_{tgt_lang}"
    # print("getting config", config)
    data = load_dataset("google/smol", config)
    for item in data['train']:
        item_src = item['src']
        item_tgts = item['trgs']
        assert isinstance(item_src, str)
        assert isinstance(item_tgts, list)
        for item_tgt in item_tgts:
            src.append(item_src)
            tgt.append(item_tgt)
    pairs = list(zip(src, tgt))
    print(f"{src_lang}-{tgt_lang}: {len(pairs)} GATITOS Pairs")
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
    print("--------------------------------------")
    print("###### download_smol.py ######")
    print("--------------------------------------")
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    get_data(langs=langs)
