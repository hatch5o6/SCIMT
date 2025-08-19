import argparse
import os
from tqdm import tqdm

"""
This counts the number of parallel sentences used, as created by the train_SC pipeline, to detect cognates.
Just counts the number of sents in the COGNATE_TRAIN/{lang}_{SC_MODEL_TYPE}-{ID}_S-{SEED}/cognate/train.{lang1} file
Will also assert that all dirs (for a lang) have the same parallel sentences for detecting cognates / makes it go slow, but a good check
Not fancy, run each time for numbers to get most up-to-date results
"""

def log(
    langs,
    COGNATE_TRAIN
):
    lang_data = {}
    for lang in langs:
        print("lang", lang)
        split_lang = [x.strip() for x in lang.split("-")]
        print("split_lang", split_lang)
        assert len(split_lang) == 2
        src, tgt = tuple(split_lang)
        assert src != ""
        assert tgt != ""
        for d in tqdm(os.listdir(COGNATE_TRAIN)):
            # print("-------------")
            d_path = os.path.join(COGNATE_TRAIN, d)
            assert os.path.isdir(d_path)

            if d.startswith(f"{lang}_"):
                d_cognate = os.path.join(d_path, "cognate")
                src_train = os.path.join(d_cognate, f"train.{src}")
                tgt_train = os.path.join(d_cognate, f"train.{tgt}")
                # print("src:", src_train)
                # print("tgt:", tgt_train)
                
                src_lines = read_f(src_train)
                tgt_lines = read_f(tgt_train)
                assert len(src_lines) == len(tgt_lines)
                pairs = list(zip(src_lines, tgt_lines))

                if lang in lang_data:
                    assert lang_data[lang] == pairs
                else:
                    lang_data[lang] = pairs
    

    print("PARALLEL SENTS FOR COGNATE DETECTION:")
    for lang, lang_pairs in lang_data.items():
        if lang_pairs is None:
            print(f"{lang} lang has no pairs?")
        print(f"{lang}: {len(lang_pairs)}")

def read_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="bho-hi,bn-as,cs-hsb,djk-en,es-an,ewe-fon,fon-ewe,fr-mfe,hi-bho,lua-bem", help="comma-delimited")
    parser.add_argument("--COGNATE_TRAIN", default="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    log(langs, args.COGNATE_TRAIN)