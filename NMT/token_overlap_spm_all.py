import argparse
import yaml
import os
import shutil
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime

from token_overlap_spm import calc_overlap

RE_FINETUNE     = r'FINETUNE.[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_FINETUNE_SC  = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN     = r'PRETRAIN.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN_SC  = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'

RE_VALID_DIR = r'[a-z]{2,3}-[a-z]{2,3}'

def main(configs_dir, out_dir):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M")
    out_dir += "_" + dt_string
    recreate_dir(out_dir)

    dict_scores = {"name": [], "JSD": [], "Jaccard": []}
    latex_path = os.path.join(out_dir, "scores_latex_table.txt")
    assert not os.path.exists(latex_path)
    for d in tqdm(sorted(os.listdir(configs_dir))):
        if not re.fullmatch(RE_VALID_DIR, d): continue
        print(f"\n############################## {d} ##############################")
        d_path = os.path.join(configs_dir, d)

        finetune_config     = read_yaml(get_exact_re_file(d_path, RE_FINETUNE))
        finetune_sc_config  = read_yaml(get_exact_re_file(d_path, RE_FINETUNE_SC))
        pretrain_config     = read_yaml(get_exact_re_file(d_path, RE_PRETRAIN))
        pretrain_sc_config  = read_yaml(get_exact_re_file(d_path, RE_PRETRAIN_SC))

        print("Normal scores -------------------------------------")
        finetune_train = finetune_config["train_data"]
        finetune_spm = finetune_config["spm"]

        pretrain_train = pretrain_config["train_data"]
        pretrain_spm = pretrain_config["spm"]
        assert pretrain_spm == finetune_spm, f"Non-matching spms for pretrain and finetune: `{pretrain_spm}`, `{finetune_spm}`"

        normal_path, normal_name = get_out_f(pretrain_config, finetune_config, out_dir)
        jsd_score, jaccard_score_result = calc_overlap(
            data1_f=pretrain_train,
            spm1=pretrain_spm,
            sc_model_id_1=None,

            data2_f=finetune_train,
            spm2=finetune_spm,
            sc_model_id_2=None,

            is_parallel=False,
            out_f=normal_path
        )
        dict_scores = add_to_dict_scores(dict_scores, normal_name, jsd_score, jaccard_score_result)

        print("SC scores -------------------------------------")
        finetune_sc_train = finetune_sc_config["train_data"]
        finetune_sc_spm = finetune_sc_config["spm"]
        finetune_sc_model_id = finetune_sc_config["sc_model_id"]

        pretrain_sc_train = pretrain_sc_config["train_data"]
        pretrain_sc_spm = pretrain_sc_config["spm"]
        pretrain_sc_model_id = pretrain_sc_config["sc_model_id"]
        assert pretrain_sc_spm == finetune_sc_spm, f"Non-matching spms for pretrain_sc and finetune_sc: `{pretrain_sc_spm}`, `{finetune_sc_spm}`"

        sc_path, sc_name = get_out_f(pretrain_sc_config, finetune_sc_config, out_dir, tag="SC")
        sc_jsd_score, sc_jaccard_score_result = calc_overlap(
            data1_f=pretrain_sc_train,
            spm1=pretrain_sc_spm,
            sc_model_id_1=pretrain_sc_model_id,

            data2_f=finetune_sc_train,
            spm2=finetune_sc_spm,
            sc_model_id_2=finetune_sc_model_id,

            is_parallel=False,
            out_f=sc_path
        )
        dict_scores = add_to_dict_scores(dict_scores, sc_name, sc_jsd_score, sc_jaccard_score_result)
    
    df_scores = pd.DataFrame(dict_scores)
    df_scores.to_latex(bug=latex_path)

def add_to_dict_scores(dict_scores, name, jsd, jaccard):
    assert sorted(list(dict_scores.keys())) == ["JSD", "Jaccard", "name"]
    assert len(dict_scores["name"]) == len(dict_scores["JSD"]) == len(dict_scores["Jaccard"])
    dict_scores["name"].append(name)
    dict_scores["JSD"].append(jsd)
    dict_scores["Jaccard"].append(jaccard["global"])
    assert len(dict_scores["name"]) == len(dict_scores["JSD"]) == len(dict_scores["Jaccard"])
    return dict_scores

def get_out_f(pretrain_config, finetune_config, out_dir, tag=""):
    f_name = f"{pretrain_config['src']}-{pretrain_config['tgt']}>>{finetune_config['src']}-{finetune_config['tgt']}"
    if tag:
        f_name += "_" + tag
    f_path = os.path.join(out_dir, f_name)
    return f_path, f_name

def recreate_dir(d):
    if os.path.exists(d):
        print("Deleting dir:", d)
        shutil.rmtree(d)
    print("Creating dir:", d)
    os.mkdir(d)

def get_exact_re_file(d_path, regex):
    matches = []
    for f in os.listdir(d_path):
        f_path = os.path.join(d_path, f)
        assert os.path.isfile(f_path), f"`{f_path}` is not a file!"
        if re.fullmatch(regex, f) != None:
            matches.append(f_path)
    assert len(matches) == 1, f"Found more than one file matching r'{regex}' in {d_path}!"
    return matches[0]

def read_yaml(f):
    print(f"reading yaml: `{f}`")
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("-o", "--out_dir", default="/home/hatch5o6/nobackup/archive/CognateMT/vocab_overlap")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    return args

if __name__ == "__main__":
    print("############################")
    print("# token_overlap_spm_all.py #")
    print("############################")
    args = get_args()
    main(args.configs_dir, args.out_dir)
