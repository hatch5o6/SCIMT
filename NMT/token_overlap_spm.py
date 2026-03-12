import argparse
from tqdm import tqdm
import json
import torch
from torchmetrics.regression import JensenShannonDivergence
from scipy.spatial.distance import jensenshannon
from collections import Counter
import csv
JSD = JensenShannonDivergence()

from spm_tokenizers import SPMTokenizer

def read_data(f):
    assert f.endswith(".txt")
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

def read_csv(f, sc_model_id=None, read_tgt=False):
    with open(f) as inf:
        lines = [l for l in csv.reader(inf)]
    assert len(lines) == 2
    assert lines[0] == ["src_lang","tgt_lang","src_path","tgt_path"]
    src_lang, tgt_lang, src_path, tgt_path = lines[1]
    if sc_model_id == None:
        assert "SC_{SC_MODEL_ID}_" not in src_path, f"SC tag IS in src_path of csv: `{f}`, but shoudln't be"
    else:
        assert "SC_{SC_MODEL_ID}_" in src_path, f"SC tag IS NOT in src_path of csv: `{f}`"
        src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)
        src_lang += f"_{sc_model_id}"
    print(f"reading sents from `{src_path}`")
    if read_tgt:
        sents = read_data(tgt_path)
        return sents, tgt_lang
    else:
        sents = read_data(src_path)
        return sents, src_lang


def calc_overlap(
    data1_f,
    spm1,
    sc_model_id_1,

    data2_f,
    spm2,
    sc_model_id_2,

    is_parallel,
    out_f,

    read_tgt1=False,
    read_tgt2=False,
):
    print("calc_overlap(")
    print(f"\tdata1_f=`{data1_f}`")
    print(f"\tspm1=`{spm1}`")
    print(f"\tsc_model_id_1=`{sc_model_id_1}`\n")

    print(f"\tdata2_f=`{data2_f}`")
    print(f"\tspm2=`{spm2}`")
    print(f"\tsc_model_id_2=`{sc_model_id_2}`\n")

    print("\tis_parallel=`{is_parallel}`")
    print("\tout_f=`{out_f}`")
    print(")")

    data1, _ = read_csv(data1_f, sc_model_id=sc_model_id_1, read_tgt=read_tgt1)
    print("data1: ", type(data1))
    # print(data1[:5])
    data2, _ = read_csv(data2_f, sc_model_id=sc_model_id_2, read_tgt=read_tgt2)
    
    tokenizer1 = SPMTokenizer(spm_name=spm1)
    tokenizer2 = SPMTokenizer(spm_name=spm2)

    print(f"Tokenizing {data1_f} ({len(data1)} seqs)\n\twith {spm1}")
    data1_toks = [
        tokenizer1.tokenize(seq)[1]
        for seq in data1
    ]
    # for i in range(5):
    #     if i < len(data1_toks):
    #         print(data1_toks[i])

    print(f"Tokenizing {data2_f} ({len(data2)} seqs)\n\twith {spm2}")
    data2_toks = [
        tokenizer2.tokenize(seq)[1]
        for seq in data2
    ]
    # for i in range(5):
    #     if i < len(data2_toks):
    #         print(data2_toks[i])

    jsd_score = calc_JSD(data1_toks, data2_toks)
    print("JSD:", type(jsd_score), jsd_score)
    out_f_jsd = out_f.replace(".json", ".JSD.json")
    with open(out_f_jsd, "w") as outf:
        outf.write(json.dumps(jsd_score, ensure_ascii=False, indent=2))

    jaccard_score_result = calc_jaccard_overlap(data1_toks, data2_toks, is_parallel=is_parallel)
    print("JACCARD:", type(jaccard_score_result), jaccard_score_result["global"])
    out_f_jacc = out_f.replace(".json", ".JACC.json")
    with open(out_f_jacc, "w") as outf:
        outf.write(json.dumps(jaccard_score_result, ensure_ascii=False, indent=2))
    
    return jsd_score, jaccard_score_result

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
        sum_sent_scores = 0
        for i, seq1 in tqdm(enumerate(data1_toks), total=len(data1_toks)):
            seq2 = data2_toks[i]
            assert i not in scores
            scores[i] = jaccard_score(seq1, seq2)
            sum_sent_scores += scores[i]

        avg_sent_score = sum_sent_scores / len(data1_toks)
        scores["avg_sent_jaccard"] = avg_sent_score
    
    # print(scores)
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
    parser.add_argument("--spm1")
    parser.add_argument("--sc_model_id_1", default=None)

    parser.add_argument("--data2")
    parser.add_argument("--spm2")
    parser.add_argument("--sc_model_id_2", default=None)

    parser.add_argument("--is_parallel", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("########################")
    print("# token_overlap_spm.py #")
    print("########################")
    args = get_args()
    calc_overlap(
        data1_f=args.data1,
        spm1=args.spm1,
        sc_model_id_1=args.sc_model_id_1,

        data2_f=args.data2,
        spm2=args.spm2,
        sc_model_id_2=args.sc_model_id_2,

        is_parallel=args.is_parallel,
        out_f=args.out
    )


