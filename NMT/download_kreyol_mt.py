import argparse
import os
import shutil
from tqdm import tqdm
import json
from datasets import load_dataset

DIVISIONS = {"train": "train", "validation": "val", "test": "test"}

def get_data(src_langs, tgt_langs, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    print("getting all possible pairs")
    pairs = []
    for src in src_langs:
        for tgt in tgt_langs:
            pair = f"{src}-{tgt}"
            pairs.append(pair)

    print("Loading and writing data")
    got_pairs_ct = 0
    manifest_f = os.path.join(out_dir, "manifest.json")
    with open(manifest_f) as inf:
        manifest = json.load(inf)
    for pair in tqdm(pairs):
        if pair in manifest: continue
        print("Downloading", pair)
        try:
            pair_data = load_dataset("jhu-clsp/kreyol-mt", pair)
            write_data(pair_data, pair, out_dir)
            got_pairs_ct += 1
        except Exception as e:
            print("COULD NOT GET PAIR", pair)
            print(e)

    with open(manifest_f) as inf:
        manifest = json.load(inf)
    sorted_manifest = {}
    keys = sorted(list(manifest.keys()))
    for key in keys:
        sorted_manifest[key] = manifest[key]
    with open(manifest_f, "w") as outf:
        outf.write(json.dumps(sorted_manifest, ensure_ascii=False, indent=2))
    
    print(f"Retrieved data for {got_pairs_ct} pairs")
    

def write_data(data, pair, out_dir):
    manifest_f = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_f):
        with open(manifest_f) as inf:
            manifest = json.load(inf)
    else:
        manifest = {}

    src_lang, tgt_lang = tuple(pair.split("-"))
    pair_dir = os.path.join(out_dir, pair)
    if os.path.exists(pair_dir):
        print("removing", pair_dir)
        shutil.rmtree(pair_dir)
    print("creating", pair_dir)
    os.mkdir(pair_dir)

    for div in DIVISIONS:
        if div not in data: continue
        div_data = data[div]['translation']
        src_f = f"{DIVISIONS[div]}.{src_lang}-{tgt_lang}.{src_lang}"
        src_f = os.path.join(pair_dir, src_f)
        tgt_f = f"{DIVISIONS[div]}.{src_lang}-{tgt_lang}.{tgt_lang}"
        tgt_f = os.path.join(pair_dir, tgt_f)
        
        if pair not in manifest:
            manifest[pair] = {}
        manifest[pair][DIVISIONS[div]] = len(div_data)

        print(f"Writing {pair} {div} data")
        with open(src_f, "w") as sf, open(tgt_f, "w") as tf:
            for segment in div_data:
                assert segment["src_lang"] == src_lang
                assert segment["tgt_lang"] == tgt_lang
                sf.write(segment["src_text"].strip() + "\n")
                tf.write(segment["tgt_text"].strip() + "\n")
    
    with open(manifest_f, "w") as outf:
        outf.write(json.dumps(manifest, ensure_ascii=False, indent=2))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="djk,gcf,gcr,hat,jam,kea,mart1259,mfe,pap,srn,tpi", help="comma-delimited list of src langs")
    parser.add_argument("--tgt", default="eng,fra,por,spa,deu,hat", help="comma-delimited list of target langs")
    parser.add_argument("--out", default="/home/hatch5o6/nobackup/archive/data/Kreyol-MT", help="directory to save data to")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("-----------------------------------")
    print("###### download_keryol_mt.py ######")
    print("-----------------------------------")
    args = get_args()
    src_langs = [l.strip() for l in args.src.split(",")]
    tgt_langs = [l.strip() for l in args.tgt.split(",")]
    get_data(
        src_langs=src_langs,
        tgt_langs=tgt_langs,
        out_dir=args.out
    )


