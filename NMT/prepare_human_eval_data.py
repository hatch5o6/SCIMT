import argparse
import csv
import json
import os
import shutil
import re
import yaml
import random

RE_FINETUNE     = r'FINETUNE.[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_FINETUNE_SC  = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN     = r'PRETRAIN.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN_SC  = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'

RE_1SET_FINETUNE_SC = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'
RE_1SET_NMT         = r'NMT.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_NMT_SC      = r'NMT.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_PRETRAIN_SC = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'



def main(
    lang_pair,
    config_dir,
    sample_size=100,
    seed=42
):
    random.seed(seed)

    dir_path = os.path.join(config_dir, lang_pair)

    finetune_cfg_f = get_exact_re_file(dir_path, RE_FINETUNE)
    finetune_cfg = read_yaml(finetune_cfg_f)

    finetune_sc_cfg_f = get_exact_re_file(dir_path, RE_FINETUNE_SC)
    finetune_sc_cfg = read_yaml(finetune_sc_cfg_f)

    assert finetune_cfg["test_data"] == finetune_sc_cfg["test_data"]
    test_csv = finetune_cfg["test_data"]

    (src_lang, 
     src_sents, 
     src_time, 
     ref_lang, 
     ref_sents, 
     ref_time) = read_csv(test_csv)
    
    assert src_lang == finetune_sc_cfg["src"] == finetune_cfg["src"]
    assert ref_lang == finetune_sc_cfg["tgt"] == finetune_cfg["tgt"]
    
    (best_hyp_lines, 
     best_hyp_file, 
     best_checkpoint) = get_predictions(finetune_cfg)
    
    (best_sc_hyp_lines,
     best_sc_hyp_file,
     best_sc_checkpoint) = get_predictions(finetune_sc_cfg)
    # sample 100 sentences
    # TODO find mt eval doc to remember how to arrange input file for mt eval :)

    assert len(src_sents) == len(ref_sents) == len(best_hyp_lines) == len(best_sc_hyp_lines)
    data = list(zip(ref_sents, best_hyp_lines, best_sc_hyp_lines))
    data_with_src = list(zip(src_sents, ref_sents, best_hyp_lines, best_sc_hyp_lines))

    assert len(data) == len(data_with_src)
    sampled_idcs = random.sample(range(len(data)), sample_size)
    data = [data[i] for i in sampled_idcs]
    data_with_src = [data_with_src[i] for i in sampled_idcs]
    assert_samples_are_same(data, data_with_src)

    human_eval_dir, ft_name, ft_sc_name = get_human_eval_dir(finetune_cfg, finetune_sc_cfg)

    data = [["ref", ft_name, ft_sc_name]] + list_of_lists(data)
    data_with_src = [["src", "ref", ft_name, ft_sc_name]] + list_of_lists(data_with_src)

    data_f = os.path.join(human_eval_dir, "just_ref.tsv")
    data_with_src_f = os.path.join(human_eval_dir, "with_src.tsv")

    write_tsv(data, data_f)
    write_tsv(data_with_src, data_with_src_f)

def get_human_eval_dir(baseline_cfg, exp_cfg):
    baseline_save_dir = get_save_dir(baseline_cfg)
    exp_save_dir = get_save_dir(exp_cfg)

    baseline_parent = parent_dir(baseline_save_dir)
    exp_parent = parent_dir(exp_save_dir)
    assert baseline_parent == exp_parent

    human_eval_dir = os.path.join(baseline_parent, "mt_eval")
    if os.path.exists(human_eval_dir):
        print("DELETING", human_eval_dir)
        shutil.rmtree(human_eval_dir)
    print("CREATING", human_eval_dir)
    os.mkdir(human_eval_dir)

    notes_f = os.path.join(human_eval_dir, "notes")
    with open(notes_f, "w") as outf:
        outf.write(f"MT Eval Comparing:\n\t-`{baseline_save_dir}`\n\t-`{exp_save_dir}`\n")

    baseline_name = baseline_save_dir.split("/")[-1]
    exp_name = exp_save_dir.split("/")[-1]

    return human_eval_dir, baseline_name, exp_name


def parent_dir(d):
    return "/".join(d.split("/")[:-1])

def assert_samples_are_same(data, data_with_src):
    assert len(data) == len(data_with_src)
    for i in range(len(data)):
        data_seq = data[i]
        data_with_src_seq = data_with_src[i]
        
        assert len(data_seq) == 3
        assert len(data_with_src_seq) == 4

        assert data_seq[0] == data_with_src_seq[1]
        assert data_seq[1] == data_with_src_seq[2]
        assert data_seq[2] == data_with_src_seq[3]
    return

def list_of_lists(lst):
    return [list(t) for t in lst]

def write_tsv(rows, path):
    with open(path, "w") as outf:
        writer = csv.writer(outf, delimiter="\t")
        writer.writerows(rows)

def get_save_dir(cfg):
    return cfg["save"] + f"_TRIAL_s={cfg['seed']}"

def get_predictions(cfg):
    save_dir = get_save_dir(cfg)
    predictions_dir = os.path.join(save_dir, "predictions")
    all_scores_f = os.path.join(predictions_dir, "all_scores.json")
    print("all_scores_f:", all_scores_f)
    all_scores = read_json(all_scores_f)
    best_checkpoint = all_scores["BEST_VAL_BLEU_CHECKPOINT"]["checkpoint"]
    print("best_checkpoint:", best_checkpoint)
    checkpoint_name = best_checkpoint.split("/")[-1]
    test_preds_f = os.path.join(predictions_dir, checkpoint_name, "test_predictions.txt")
    print("hyp_file:", test_preds_f)
    test_lines = read_data(test_preds_f)
    return test_lines, test_preds_f, best_checkpoint

def read_json(f):
    with open(f) as inf:
        data = json.load(inf)
    return data

def read_data(f):
    assert f.endswith(".txt")
    with open(f) as inf:
        data = [l.strip() for l in inf.readlines()]
    return data

def read_csv(f, sc_model_id=None):
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
    
    tgt_time = os.path.getmtime(tgt_path)
    src_time = os.path.getmtime(src_path)

    src_sents = read_data(src_path)
    tgt_sents = read_data(tgt_path)

    return src_lang, src_sents, src_time, tgt_lang, tgt_sents, tgt_time

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
    parser.add_argument("--lang_pair", "-l")
    parser.add_argument("--nmt_config_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        lang_pair=args.lang_pair, 
        config_dir=args.nmt_config_dir, 
        sample_size=args.sample_size, 
        seed=args.seed)