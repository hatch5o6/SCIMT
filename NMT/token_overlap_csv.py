import argparse
from tqdm import tqdm
import json
import csv
import torch
from torchmetrics.regression import JensenShannonDivergence
from scipy.spatial.distance import jensenshannon
from collections import Counter
JSD = JensenShannonDivergence()

from torch.utils.data import DataLoader
from parallel_datasets import MultilingualDataset
from parallel_datasets_sc import SCAlignedMultilingualDataset
from spm_tokenizers import SPMTokenizer
from sc_aligned_spm_tokenizers import SCAlignedSPMTokenizer


def calc_overlap(
    pl,
    cl,
    tl,

    data_csv,
    sc_csv,
    sc_model_id,

    spm1,
    spm1_sc,
    spm1_type,

    spm2,
    spm2_sc,
    spm2_type,

    VOCAB_SIZE_CAP,
    out_f
):
    ## PRINTING ARGS ##
    print("--------------CALC OVERLAP--------------")
    print(f"data_csv: `{data_csv}`")
    print(f"sc_csv: `{sc_csv}`")
    print(f"sc_model_id: `{sc_model_id}`")

    print(f"spm1: `{spm1}`")
    print(f"spm1_sc: `{spm1_sc}`")
    print(f"spm1_type: `{spm1_type}`")

    print(f"spm2: `{spm2}`")
    print(f"spm2_sc: `{spm2_sc}`")
    print(f"spm2_type: `{spm2_type}`")

    print(f"VOCAB_SIZE_CAP: `{VOCAB_SIZE_CAP}`")
    print(f"out_f: `{out_f}`")
    print("----------------------------------------")
    #####

    assert spm1_type in ["spm", "sc_aligned"]
    assert spm2_type in ["spm", "sc_aligned"]

    csv_asserts(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE=False)
    csv_asserts(sc_csv, pl, cl, tl, sc_model_id, IS_SC_FILE=True)

    plain_dataloader=get_sc_dataloader(
        data_csv=data_csv, 
        shuffle=True
    )
    sc_dataloader=get_sc_dataloader(
        data_csv=data_csv,
        sc_data_csv=sc_csv,
        sc_model_id=sc_model_id,
        shuffle=True
    )


def get_sc_dataloader(
    data_csv,
    sc_data_csv=None,
    sc_model_id=None,
    upsample=False,
    shuffle=False,
    PRINT:int=None
):
    assert isinstance(PRINT, int) or PRINT == None
    print("SC DATA:")
    # SHOULD ONLY DO THIS ON TRAINING PROBABLY
    # Upsample=False and shuffle=True used for training data in pretrain -> finetune scenarios.
    sc_dataset = SCAlignedMultilingualDataset(
        data_csv=data_csv,
        sc_data_csv=sc_data_csv,
        sc_model_id=sc_model_id,
        append_src_lang_tok=False,
        append_tgt_lang_tok=False,
        append_tgt_to_src=False,
        upsample=upsample,
        shuffle=shuffle
    )
    sc_dataloader = DataLoader(
        sc_dataset,
        batch_size=1024,
        shuffle=False
    )

    if isinstance(PRINT, int):
        i = 0
        for src_tags, src_lines, sc_lines, tgt_tags, tgt_lines in sc_dataloader:
            batch = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
            for src_tag, src_line, sc_line, tgt_tag, tgt_line in batch:
                print(f"--------------- {i} -----------------")
                print(f"(src) {src_tag}: `{src_line}`")
                print(f"(sc)  {src_tag}: `{sc_line}`")
                print(f"(tgt) {tgt_tag}: `{tgt_line}`")
                i += 1
            
            if i >= PRINT:
                break
        return None
    else:
        return sc_dataloader

def csv_asserts(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE=False):
    with open(data_csv, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    rows = rows[1:]

    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    for src_lang, tgt_lang, src_path, tgt_path in rows:
        assert tgt_lang == tl
        assert "SC_{SC_MODEL_ID}" not in tgt_path

        assert src_lang in [pl, cl]
        if IS_SC_FILE and src_lang == pl:
            assert "SC_{SC_MODEL_ID}" in src_path
            assert src_path.endswith(f"SC_{SC_MODEL_ID}_{pl}2{cl}.txt")
            # assert sc_model_id is not None
        else:
            assert "SC_{SC_MODEL_ID}" not in src_path


# def read_data(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE=False):
#     assert_model_id(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl", help="parent language")
    parser.add_argument("--cl", help="child language")
    parser.add_argument("--tl", help="target language")

    parser.add_argument("--data_csv") # should be augmented traininng csv, and only have two lang pairs (e.g. mfe-en, fr-en)
    parser.add_argument("--sc_csv") # should be sc version of data_csv
    parser.add_argument("--sc_model_id")

    # for first of lang pairs in data_csv
    parser.add_argument("--spm1")
    parser.add_argument("--spm1_sc")
    parser.add_argument("--spm1_type", choices=["spm", "sc_aligned"])

    # for second of lang pairs in data_csv
    parser.add_argument("--spm2")
    parser.add_argument("--spm2_sc")
    parser.add_argument("--spm2_type", choices=["spm", "sc_aligned"])

    # parser.add_argument("--is_parallel", action="store_true")
    parser.add_argument("--VOCAB_SIZE_CAP", type=int, default=32000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("----------------------------------")
    print("###### token_overlap_csv.py ######")
    print("----------------------------------")
    args = get_args()
    calc_overlap(
        pl=args.pl,
        cl=args.cl,
        tl=args.tl,

        data_csv=args.data_csv,
        sc_csv=args.sc_csv,
        sc_model_id=args.sc_model_id,

        spm1=args.spm1,
        spm1_sc=args.spm1_sc,
        spm1_type=args.spm1_type,

        spm2=args.spm2,
        spm2_sc=args.spm2_sc,
        spm2_type=args.spm2_type,

        VOCAB_SIZE_CAP=args.VOCAB_SIZE_CAP,
        out_f=args.out
    )
    