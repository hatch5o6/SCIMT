import argparse
from tqdm import tqdm
import json
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
    data_csv,
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


def assert_model_id(data_csv, sc_model_id):
    with open(data_csv) as inf:
        lines = [l.strip() for l in inf.readlines()]
    assert "SC_{SC_MODEL_ID}" not in lines[0] # not in header
    assert "SC_{SC_MODEL_ID}" not in lines[1] # not in first line (child language)
    if "SC_{SC_MODEL_ID}" in lines[2]: # if in second line (parent language)
        assert sc_model_id is not None
    else:
        assert sc_model_id is None


def read_data(data_csv, sc_model_id):
    assert_model_id(data_csv, sc_model_id)
    



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv") # should be augmented traininng csv, and only have two lang pairs (e.g. mfe-en, fr-en)
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
    print("------------------------------")
    print("###### token_overlap_csv.py ######")
    print("------------------------------")
    args = get_args()
    calc_overlap(
        data_csv=args.data_csv,
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
    