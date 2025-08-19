"""
Asserts data in all versions of a SC .cfg config are the same
"""
import argparse
import os
from tqdm import tqdm

def check(
    langs,
    standard,
    COPPERMT
):
    CopperMT_lang_dirs = {
        l: read_COPPERMT_lang_dir(l, COPPERMT)
        for l in langs
    }

    standard_dirs = {
        l: read_standard_dirs(l, standard)
        for l in langs
    }

    NOT_EQUAL = []
    TOTAL = 0
    for lang, lang_dirs in CopperMT_lang_dirs.items():
        print("CHECKING LANG", lang)

        standard_dir = standard_dirs.get(lang, None)
        if standard_dir is None:
            standard_dir = lang_dirs[0]
        
        for lang_dir in tqdm(lang_dirs):
            for f in os.listdir(standard_dir):
                TOTAL += 1
                standard_f = os.path.join(standard_dir, f)
                copper_f = os.path.join(lang_dir, f)
                print("COMPARING:")
                print(f"\tSTANDARD: `{standard_f}`")
                print(f"\tCOPPERMT: `{copper_f}`")
                standard_content = read_f(standard_f)
                copper_content = read_f(copper_f)
                are_equal = standard_content == copper_content
                print(f"\t", are_equal)
                if not are_equal:
                    NOT_EQUAL.append((standard_f, copper_f))
        
    print("NOT_EQUAL_FS:")
    for sf, cf in NOT_EQUAL:
        print(f"-----------------------------------------")
        print(f"standard: `{sf}`")
        print(f"coppermt: `{cf}`")

    print("NOT_EQUAL:", len(NOT_EQUAL))
    print("TOTAL:", TOTAL)

def read_f(f):
    with open(f) as inf:
        content = inf.read()
    return content

def read_standard_dirs(lang, standard_dir):
    lang_name = lang.replace("-", "_")
    standard_dir = os.path.join(standard_dir, lang_name, "0") 
    if os.path.exists(standard_dir):
        return standard_dir
    else:
        return None

def read_COPPERMT_lang_dir(lang, COPPERMT):
    data_dirs = []
    lang_name = lang.replace("-", "_")
    for d in os.listdir(COPPERMT):
        if d.startswith(lang_name + "_"):
            d_path = os.path.join(COPPERMT, d)
            assert os.path.isdir(d_path)
            data_dir = os.path.join(d_path, f"inputs/split_data/{lang_name}/0")
            data_dirs.append(data_dir)
    return data_dirs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="bho-hi,bn-as,cs-hsb,djk-en,es-an,ewe-fon,fon-ewe,fr-mfe,hi-bho,lua-bem", help="comma-delimited list")
    parser.add_argument("--standard", default="/home/hatch5o6/nobackup/archive/data/CognateSplits", help="directory with data to use as the standard")
    parser.add_argument("--COPPERMT", default="/home/hatch5o6/nobackup/archive/CopperMT", help="CopperMT directory -- will make sure data in each model folder matches respective language data in --standard")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    check(
        langs=langs, 
        standard=args.standard, 
        COPPERMT=args.COPPERMT
    )