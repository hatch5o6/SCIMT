import argparse
import os
import yaml
import csv
import json
import re
from tqdm import tqdm
from datetime import datetime
import Levenshtein
import statistics
import numpy as np

RE_FINETUNE     = r'FINETUNE.[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_FINETUNE_SC  = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN     = r'PRETRAIN.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN_SC  = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'

RE_1SET_FINETUNE_SC = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'
RE_1SET_NMT         = r'NMT.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_NMT_SC      = r'NMT.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_PRETRAIN_SC = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'

def main(
    config_dirs,
    out_dir,
    langs
):
    for d in tqdm(os.listdir(config_dirs)):
        if "_chosen" in d: continue
        if d == "_archive": continue
        d_path = os.path.join(config_dirs, d)
        if not os.path.isdir(d_path): continue
        if langs and d not in langs: continue
        if d in ["djk-en", "bho-hi"]: continue #TODO need to develop the metric for ONE-SET scenarios
        print(f"########################## {d} ###########################")
        calc(nmt_config_dir=d_path, out_dir=out_dir)


def calc(
    nmt_config_dir,
    out_dir
):
    pretrain_cfg_path = get_exact_re_file(nmt_config_dir, RE_PRETRAIN)
    pretrain_cfg = read_yaml(pretrain_cfg_path)

    pretrain_sc_cfg_path = get_exact_re_file(nmt_config_dir, RE_PRETRAIN_SC)
    pretrain_sc_cfg = read_yaml(pretrain_sc_cfg_path)

    src_sents, src_lang, norm_src_time, norm_tgt_time = read_csv(pretrain_cfg["train_data"])
    src_sc_sents, src_sc_lang, sc_src_time, sc_tgt_time = read_csv(pretrain_sc_cfg["train_data"], sc_model_id=pretrain_sc_cfg["sc_model_id"])


    scores = get_diffs(src_sents, src_sc_sents)
    if scores["total_diff_len_sents"] == 0:
        scores["word_scores"] = get_word_diff(src_sents, src_sc_sents)

    scores["cfg"] = pretrain_cfg_path
    scores["cfg_sc"] = pretrain_sc_cfg_path
    scores["lang1"] = src_lang
    scores["lang2"] = src_sc_lang
    scores["norm_times"] = " ; ".join([str(norm_src_time), str(norm_tgt_time)])
    scores["sc_times"] = " ; ".join([str(sc_src_time), str(sc_tgt_time)])
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    scores_out_f = os.path.join(out_dir, f"{src_lang}-{src_sc_lang}'.{timestamp}.json")
    write_scores(scores, scores_out_f)

def write_scores(scores, out_f):
    with open(out_f, "w") as outf:
        outf.write(json.dumps(scores, ensure_ascii=False, indent=2))

def nled(seq1, seq2):
    max_len = max(len(seq1), len(seq2))
    distance = Levenshtein.distance(seq1, seq2)
    normalized = distance / max_len
    return normalized

def get_diffs(sents1, sents2):
    if len(sents1) != len(sents2):
        raise ValueError(f"len sents1 ({len(sents1)}) is different than len sents2 ({len(sents2)})")
    
    diff_len_sents = []
    nleds = []
    for i in range(len(sents1)):
        s1 = sents1[i]
        s2 = sents2[i]
        words1 = s1.split()
        words2 = s2.split()
        if len(words1) != len(words2):
            diff_len_sents.append(i)
        nled_value = nled(s1, s2)
        nleds.append(nled_value)
    assert len(nleds) == len(sents1) == len(sents2)
    avg_nled = sum(nleds) / len(nleds)
    med_nled = statistics.median(nleds)
    hist, edges = np.histogram(nleds, bins=50)
    hist = hist.tolist()
    edges = edges.tolist()
    hist_per = [num / len(sents1) for num in hist]

    metrics = {
        "total": len(sents1),
        "avg_nled": avg_nled,
        "med_nled": med_nled,
        "hist": hist,
        "hist_per": hist_per,
        "edges": edges,
        "total_diff_len_sents": len(diff_len_sents),
        "diff_len_sents": diff_len_sents
    }
    return metrics
        

def get_word_diff(sents1, sents2):
    words1 = get_words(sents1)
    words2 = get_words(sents2)
    if len(words1) != len(words2):
        raise ValueError(f"words1 ({len(words1)}) has different length than words2 ({len(words2)})")
    
    # pairs = list(zip(words1, words2))
    diff = 0
    diff_word_nleds = []
    i = 0
    for w1 in words1:
        w2 = words2[i]
        if w1 != w2:
            nled_value = nled(w1, w2)
            diff_word_nleds.append(nled_value)
            diff += 1
        i += 1
    
    avg_diff_word_nled = sum(diff_word_nleds) / len(diff_word_nleds)
    med_diff_word_nled = statistics.median(diff_word_nleds)
    hist, edges = np.histogram(diff_word_nleds, bins=10)
    hist = hist.tolist()
    edges = edges.tolist()
    hist_per = [num / len(diff_word_nleds) for num in hist]

    metrics = {
        "total": len(words1),
        "num_different": diff,
        "ratio_different": diff / len(words1),
        "percent_different": round(diff / len(words1) * 100, 2),
        "avg_nled_of_diff_words": avg_diff_word_nled,
        "med_nled_of_diff_words": med_diff_word_nled,
        "hist_nled_of_diff_words": hist,
        "edges_nled_of_diff_words": edges,
        "hist_per_nled_of_diff_words": hist_per
    }
    return metrics
    
def get_words(sents):
    words = []
    for sent in sents:
        words += sent.split()
    return words

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
    
    tgt_time = get_mod_time(tgt_path)
    src_time = get_mod_time(src_path)

    if read_tgt:
        sents = read_data(tgt_path)
        return sents, tgt_lang, src_time, tgt_time
    else:
        sents = read_data(src_path)
        return sents, src_lang, src_time, tgt_time
    
def get_mod_time(path):
    timestamp = os.path.getmtime(path)
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%H:%M:%S")

def read_yaml(f):
    if f == None:
        return None
    print(f"reading yaml: `{f}`")
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_exact_re_file(d_path, regex):
    matches = []
    for f in os.listdir(d_path):
        f_path = os.path.join(d_path, f)
        assert os.path.isfile(f_path), f"`{f_path}` is not a file!"
        if re.fullmatch(regex, f) != None:
            matches.append(f_path)
    assert len(matches) in [0, 1], f"Found more than one file matching r'{regex}' in {d_path}: {matches}"
    if len(matches) == 1:
        return matches[0]
    else:
        print(f"Found no matches for r'{regex}' in {d_path}")
        return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nmt_config_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("-o", "--out_dir", default="/home/hatch5o6/Cognate/code/NMT/results")
    parser.add_argument("--langs", help="comma-delimited list of lang pairs")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = args.langs
    if langs:
        langs = [l.strip() for l in langs.split(",")]
    main(args.nmt_config_dir, args.out_dir, langs)
