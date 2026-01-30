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
# from sc_aligned_spm_tokenizers import SCAlignedSPMTokenizer


# def read_data(f, sc_f=None, sc_modelid=None):
#     if f.endswith(".txt"):
#         assert sc_f is None
#         assert sc_modelid is None
#         with open(f) as inf:
#             data = [line.strip() for line in inf]
#     else:
#         assert f.endswith(".csv")
#         if sc_f is not None:
#             assert sc_f.endswith(".csv")
#             assert sc_modelid is not None
            
#             # SHOULD ONLY DO THIS ON TRAINING PROBABLY
#             # Upsample=False and shuffle=True used for training data in pretrain -> finetune scenarios.
#             sc_dataset = SCAlignedMultilingualDataset(
#                 data_csv=f,
#                 sc_data_csv=sc_f,
#                 append_src_lang_tok=False,
#                 append_tgt_lang_tok=False,
#                 append_tgt_to_src=False,
#                 upsample=False,
#                 shuffle=True
#             )
#             sc_dataloader = DataLoader(
#                 sc_dataset,
#                 batch_size=100,
#                 shuffle=False
#             )

#     return data

def read_data(f):
    assert f.endswith(".txt")
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

def calc_overlap(
    data1_f,
    data1sc_f,
    data1sc_modelid,

    spm1,
    spm1_sc,
    spm1_type,

    data2_f,
    data2sc_f,
    data2sc_modelid,

    spm2,
    spm2_sc,
    spm2_type,

    is_parallel,
    VOCAB_SIZE_CAP,
    out_f
):
    ## PRINTING ARGS ##
    print("--------------CALC OVERLAP--------------")
    print(f"data1_f: `{data1_f}`")
    print(f"data1sc_f: `{data1sc_f}`")
    print(f"data1sc_modelid: `{data1sc_modelid}`")

    print(f"\nspm1: `{spm1}`")
    print(f"spm1_sc: `{spm1_sc}`")
    print(f"spm1_type: `{spm1_type}`")

    print(f"\ndata2_f: `{data2_f}`")
    print(f"data2sc_f: `{data2sc_f}`")
    print(f"data2sc_modelid: `{data2sc_modelid}`")

    print(f"\nspm2: `{spm2}`")
    print(f"spm2_sc: `{spm2_sc}`")
    print(f"spm2_type: `{spm2_type}`")

    print(f"\nis_parallel: `{is_parallel}`")
    print(f"VOCAB_SIZE_CAP: `{VOCAB_SIZE_CAP}`")
    print(f"out_f: `{out_f}`")
    print("----------------------------------------")
    #####

    assert spm1_type in ["spm", "sc_aligned"]
    assert spm2_type in ["spm", "sc_aligned"]

    data1 = read_data(data1_f, data1sc_f, data1sc_modelid)
    data2 = read_data(data2_f, data2sc_f, data2sc_modelid)
    
    if spm1_type == "spm":
        tokenizer1 = SPMTokenizer(spm_name=spm1)
    elif spm1_type == "sc_aligned":
        tokenizer1 = SCAlignedSPMTokenizer(
            fr_spm_name=spm1, 
            sc_spm_name=spm1_sc,
            VOCAB_SIZE_CAP=VOCAB_SIZE_CAP
        )

    if spm2_type == "spm":
        tokenizer2 = SPMTokenizer(spm_name=spm2)
    elif spm2_type == "sc_aligned":
        tokenizer2 = SCAlignedSPMTokenizer(
            fr_spm_name=spm2, 
            sc_spm_name=spm2_sc,
            VOCAB_SIZE_CAP=VOCAB_SIZE_CAP
        )

    # print(f"Tokenizing {data1_f}\n\twith {spm1}")
    data1_toks = [
        tokenizer1.tokenize(seq)[1]
        for seq in tqdm(data1)
    ]
    # for i in range(5):
    #     if i < len(data1_toks):
    #         print(data1_toks[i])
    # print(f"Tokenizing {data2_f}\n\twith {spm2}")
    data2_toks = [
        tokenizer2.tokenize(seq)[1]
        for seq in tqdm(data2)
    ]
    # for i in range(5):
    #     if i < len(data2_toks):
    #         print(data2_toks[i])

    jsd_score = calc_JSD(data1_toks, data2_toks)
    print("JSD:", type(jsd_score), jsd_score)
    out_f_jsd = out_f.replace(".json", ".JSD.json")
    with open(out_f_jsd, "w") as outf:
        outf.write(json.dumps(jsd_score, ensure_ascii=False, indent=2))

    jaccard_score = calc_jaccard_overlap(data1_toks, data2_toks, is_parallel=is_parallel)
    print("JACCARD:", type(jaccard_score), jaccard_score["global"])
    out_f_jacc = out_f.replace(".json", ".JACC.json")
    with open(out_f_jacc, "w") as outf:
        outf.write(json.dumps(jaccard_score, ensure_ascii=False, indent=2))


def calc_jaccard_overlap(data1_toks, data2_toks, is_parallel=False):
    print("Calculating Jaccard Similarity")
    scores = {}
    data1_set = set()
    for seq in data1_toks:
        data1_set.update(seq)
    data2_set = set()
    for seq in data2_toks:
        data2_set.update(seq)
    scores["global"] = jaccard_score(data1_set, data2_set)

    if is_parallel:
        # only for parallel data
        assert len(data1_toks) == len(data2_toks)
        for i, seq1 in tqdm(enumerate(data1_toks), total=len(data1_toks)):
            seq2 = data2_toks[i]
            assert i not in scores
            scores[i] = jaccard_score(seq1, seq2)
    
    return scores

def jaccard_score(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)

    score = len(intersection) / len(union)
    return score


def calc_JSD(data_toks1, data_toks2):
    print("Calculating Jensen-Shanon Divergence")
    raw_dist1, norm_dist1 = get_distribution(data_toks1)
    raw_dist2, norm_dist2 = get_distribution(data_toks2)
    for tok in norm_dist1:
        if tok not in norm_dist2:
            norm_dist2[tok] = 0
    for tok in norm_dist2:
        if tok not in norm_dist1:
            norm_dist1[tok] = 0
    
    all_toks = set(norm_dist1.keys())
    assert set(norm_dist2.keys()) == all_toks
    # print("UNIQUE_TOKENS", len(norm_dist1.keys()))

    sorted_dist1 = []
    sorted_dist2 = []
    for tok in sorted(list(all_toks)):
        sorted_dist1.append(norm_dist1[tok])
        sorted_dist2.append(norm_dist2[tok])

    assert len(sorted_dist1) == len(sorted_dist2)
    print("TOKS:", len(sorted_dist1))

    scipy_jsd_score = jensenshannon(sorted_dist1, sorted_dist2) ** 2
    
    # batch_size = 10
    # batched_sorted_dist1 = batch_array(sorted_dist1, batch_size=batch_size)
    # batched_sorted_dist2 = batch_array(sorted_dist2, batch_size=batch_size)

    sorted_dist1 = torch.tensor([sorted_dist1])
    sorted_dist2 = torch.tensor([sorted_dist2])

    # batched_sorted_dist1 = torch.tensor(batched_sorted_dist1)
    # batched_sorted_dist2 = torch.tensor(batched_sorted_dist2)

    jsd_score = JSD(sorted_dist1, sorted_dist2).item()
    # batched_jsd_score = JSD(batched_sorted_dist1, batched_sorted_dist2).item()

    print("JSD", jsd_score)
    print("SCIPY JSD", scipy_jsd_score)

    return scipy_jsd_score

def batch_array(alist, batch_size=64):
    batches = []
    batch = []
    for s, seq in enumerate(alist):
        if s > 0 and len(batch) == batch_size:
            batches.append(batch)
            batch = []
        batch.append(seq)
    if len(batch) > 0:
        while len(batch) < batch_size:
            batch.append(0)
        batches.append(batch)
    return batches

def get_distribution(sequences):
    cts = Counter()
    total = 0
    for seq in sequences:
        # print("SEQUENCE: ", type(seq), seq)
        for tok in seq:
            cts[tok] += 1
            total += 1
    normalized = {}
    for tok, ct in cts.items():
        assert tok not in normalized
        normalized[tok] = ct / total
    return cts, normalized


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data1")
    parser.add_argument("--data1sc")
    parser.add_argument("--data1sc_modelid")

    parser.add_argument("--spm1")
    parser.add_argument("--spm1_sc")
    parser.add_argument("--spm1_type", choices=["spm", "sc_aligned"])

    parser.add_argument("--data2")
    parser.add_argument("--data2sc")
    parser.add_argument("--data2sc_modelid")

    parser.add_argument("--spm2")
    parser.add_argument("--spm2_sc")
    parser.add_argument("--spm2_type", choices=["spm", "sc_aligned"])

    parser.add_argument("--is_parallel", action="store_true")
    parser.add_argument("--VOCAB_SIZE_CAP", type=int, default=32000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("------------------------------")
    print("###### token_overlap.py ######")
    print("------------------------------")
    args = get_args()
    calc_overlap(
        data1_f=args.data1,
        data1sc_f=args.data1sc,
        data1sc_modelid=args.data1sc_modelid,

        spm1=args.spm1,
        spm1_sc=args.spm1_sc,
        spm1_type=args.spm1_type,

        data2_f=args.data2,
        data2sc_f=args.data2sc,
        data2sc_modelid=args.data2sc_modelid,

        spm2=args.spm2,
        spm2_sc=args.spm2_sc,
        spm2_type=args.spm2_type,

        is_parallel=args.is_parallel,
        VOCAB_SIZE_CAP=args.VOCAB_SIZE_CAP,
        out_f=args.out
    )



