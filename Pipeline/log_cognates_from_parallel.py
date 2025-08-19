import argparse
import os
from tqdm import tqdm

"""
This counts the number of cognates detected in parallel data that was used in final experiments.
If the data was split, then we count sum of train + val + test (because we made the test source-side unique when splitting)
Otherwise, we just count the parallel file.
"""

def log(
    langs,
    COGNATE_TRAIN,
    NG_is_False, 
    COGNATE_THRESH, 
    SEED
):
    if NG_is_False:
        NG = False
    else:
        NG = True

    if NG == True:
        NG_tag = ".NG"
    else:
        NG_tag = ""

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
            d_path = os.path.join(COGNATE_TRAIN, d)
            assert os.path.isdir(d_path)

            if d.startswith(f"{lang}_"):
                d_fastalign = os.path.join(d_path, "fastalign")
                SPLIT_OCCURRED = splitting_occured(d_fastalign, src, tgt, NG=NG, THRESH=COGNATE_THRESH, SEED=SEED)
                if SPLIT_OCCURRED:
                    # print("SPLIT OCCURRED")
                    src_files = [
                        f"word_list.{src}-{tgt}{NG_tag}.cognates.{COGNATE_THRESH}.parallel-{src}.train-s={SEED}.txt",
                        f"word_list.{src}-{tgt}{NG_tag}.cognates.{COGNATE_THRESH}.parallel-{src}.val-s={SEED}.txt",
                        f"word_list.{src}-{tgt}{NG_tag}.cognates.{COGNATE_THRESH}.parallel-{src}.test-s={SEED}.txt"
                    ]
                else:
                    # print("NO SPLIT")
                    src_files = [
                        f"word_list.{src}-{tgt}{NG_tag}.cognates.{COGNATE_THRESH}.parallel-{src}.txt"
                    ]
                
                src_files = [os.path.join(d_fastalign, fx) for fx in src_files]
                src_lines = read_files(src_files)
                if lang in lang_data:
                    assert lang_data[lang] == {"cognates": len(src_lines), "SPLIT_OCCURED": SPLIT_OCCURRED}
                else:
                    lang_data[lang] = {"cognates": len(src_lines), "SPLIT_OCCURED": SPLIT_OCCURRED}
    
    print("COGNATES FROM PARALLEL DATA:")
    for lang, n_cognates in lang_data.items():
        print(f"\t{lang}: {n_cognates}")       

def read_files(files):
    lines = []
    for f in files:
        lines += read_f(f)
    return lines

def splitting_occured(d_fastalign, src, tgt, NG=True, THRESH=0.5, SEED=0):
    #TODO swap out threshold "0.5" for variable 
    if NG == True:
        NG_tag = ".NG"
    else:
        NG_tag = ""
    parallel_src = f"word_list.{src}-{tgt}{NG_tag}.cognates.{THRESH}.parallel-{src}.txt"
    parallel_src_path = os.path.join(d_fastalign, parallel_src)
    assert os.path.exists(parallel_src_path)

    parallel_tgt = f"word_list.{src}-{tgt}{NG_tag}.cognates.{THRESH}.parallel-{tgt}.txt"
    parallel_tgt_path = os.path.join(d_fastalign, parallel_tgt)
    assert os.path.exists(parallel_tgt_path)

    parallel_src_train_f = parallel_src_path[:-3] + f"train-s={SEED}.txt"
    if os.path.exists(parallel_src_train_f):
        SPLIT_OCCURRED = True
    else:
        SPLIT_OCCURRED = False
    return SPLIT_OCCURRED

def read_f(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    return lines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="bho-hi,bn-as,cs-hsb,djk-en,es-an,ewe-fon,fon-ewe,fr-mfe,hi-bho,lua-bem", help="comma-delimited")
    parser.add_argument("--COGNATE_TRAIN", default="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN")
    parser.add_argument("--NG_False", action="store_true", help="if doing NG=False data, pass this")
    parser.add_argument("--COGNATE_THRESH", default=0.5, help="pass in the cognate edit distance threshold that was used")
    parser.add_argument("--SEED", default=0, help="pass in the seed that was used")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    log(langs, args.COGNATE_TRAIN, args.NG_False, args.COGNATE_THRESH, args.SEED)