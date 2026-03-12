import argparse
import yaml
import os
import shutil
from tqdm import tqdm
import re
import pandas as pd
from datetime import datetime
import json

from token_overlap_spm import calc_overlap

RE_FINETUNE     = r'FINETUNE.[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_FINETUNE_SC  = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN     = r'PRETRAIN.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_PRETRAIN_SC  = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'

RE_1SET_FINETUNE_SC = r'FINETUNE.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}\>\>[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'
RE_1SET_NMT         = r'NMT.[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_NMT_SC      = r'NMT.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.yaml'
RE_1SET_PRETRAIN_SC = r'PRETRAIN.SC_[a-z]{2,3}2[a-z]{2,3}-[a-z]{2,3}.ATT.yaml'

RE_VALID_DIR = r'[a-z]{2,3}-[a-z]{2,3}'

DATA_HEAD = "Data"
JSD_HEAD = "JSD ↓"
JACCARD_HEAD = "Jaccard ↑"

def main(configs_dir, out_dir, lang_pairs):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H:%M")
    out_dir += "_" + dt_string
    recreate_dir(out_dir)

    dict_scores = {DATA_HEAD: [], JSD_HEAD: [], JACCARD_HEAD: []}
    latex_path = os.path.join(out_dir, "scores_latex_table.txt")
    latex_long_path = os.path.join(out_dir, "scores_latex_long_table.txt")
    print("will write latex file to:", latex_path)
    assert not os.path.exists(latex_path)
    for d in tqdm(sorted(os.listdir(configs_dir))):
        if not re.fullmatch(RE_VALID_DIR, d): continue
        if lang_pairs and d not in lang_pairs: continue
        print(f"\n\n\n############################## {d} ##############################")
        d_path = os.path.join(configs_dir, d)

        finetune_config     = read_yaml(get_exact_re_file(d_path, RE_FINETUNE))
        finetune_sc_config  = read_yaml(get_exact_re_file(d_path, RE_FINETUNE_SC))
        pretrain_config     = read_yaml(get_exact_re_file(d_path, RE_PRETRAIN))
        pretrain_sc_config  = read_yaml(get_exact_re_file(d_path, RE_PRETRAIN_SC))

        is_three_set = all([
            finetune_config is not None,
            finetune_sc_config is not None,
            pretrain_config is not None,
            pretrain_sc_config is not None
        ])
        
        if is_three_set:
            print("\n###THREE-SET###")
            print("\nNormal scores -------------------------------------")
            finetune_train = finetune_config["train_data"]
            finetune_spm = finetune_config["spm"]

            pretrain_train = pretrain_config["train_data"]
            pretrain_spm = pretrain_config["spm"]
            assert pretrain_spm == finetune_spm, f"Non-matching spms for pretrain and finetune: `{pretrain_spm}`, `{finetune_spm}`"

            normal_path, normal_name = get_out_f(pretrain_config["src"], finetune_config["src"], out_dir)
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

            print("\nSC scores -------------------------------------")
            finetune_sc_train = finetune_sc_config["train_data"]
            finetune_sc_spm = finetune_sc_config["spm"]
            finetune_sc_model_id = finetune_sc_config["sc_model_id"]

            assert finetune_sc_train == finetune_train
            assert finetune_sc_spm != finetune_spm

            pretrain_sc_train = pretrain_sc_config["train_data"]
            pretrain_sc_spm = pretrain_sc_config["spm"]
            pretrain_sc_model_id = pretrain_sc_config["sc_model_id"]
            assert pretrain_sc_spm == finetune_sc_spm, f"Non-matching spms for pretrain_sc and finetune_sc: `{pretrain_sc_spm}`, `{finetune_sc_spm}`"

            assert pretrain_sc_train != pretrain_train
            assert pretrain_sc_spm != pretrain_spm

            sc_path, sc_name = get_out_f(prime(pretrain_sc_config["src"]), finetune_sc_config["src"], out_dir)
            sc_jsd_score, sc_jaccard_score_result = calc_overlap(
                data1_f=pretrain_sc_train,
                spm1=pretrain_sc_spm,
                sc_model_id_1=pretrain_sc_model_id,

                data2_f=finetune_sc_train,
                spm2=finetune_sc_spm,
                # sc_model_id_2=finetune_sc_model_id,
                sc_model_id_2=None, # fine-tuning data has not had SC mappings applied

                is_parallel=False,
                out_f=sc_path
            )
            dict_scores = add_to_dict_scores(dict_scores, sc_name, sc_jsd_score, sc_jaccard_score_result)
        else:
            print("\n###ONE-SET###")
            #TODO Need to get overlap between SL and TL, SL' and TL, and TL' and SL (i.e. adds three rows, whereas above only adds 2) // actually will be four rows
            # will need a new get_out_f function to return the appropriate name
            # Names will be SL-TL, SL'-TL, and pSL-TL and pSL-TL'

            nmt_config = read_yaml(get_exact_re_file(d_path, RE_1SET_NMT))
            nmt_sc_config = read_yaml(get_exact_re_file(d_path, RE_1SET_NMT_SC))
            finetune_sc_config = read_yaml(get_exact_re_file(d_path, RE_1SET_FINETUNE_SC))
            pretrain_sc_config = read_yaml(get_exact_re_file(d_path, RE_1SET_PRETRAIN_SC))

            assert all([
                nmt_config is not None,
                nmt_sc_config is not None,
                finetune_sc_config is not None,
                pretrain_sc_config is not None
            ])

            # SL-TL
            print("\nSL-TL scores -------------------------------------")
            nmt_train = nmt_config["train_data"]
            nmt_spm = nmt_config["spm"]
            sl_tl_path, sl_tl_name = get_out_f(nmt_config["src"], nmt_config["tgt"], out_dir, tag="1")
            sl_tl_jsd_score, sl_tl_jaccard_score_result = calc_overlap(
                data1_f=nmt_train,
                spm1=nmt_spm,
                sc_model_id_1=None,

                data2_f=nmt_train,
                spm2=nmt_spm,
                sc_model_id_2=None,
                read_tgt2=True,

                is_parallel=True,
                out_f=sl_tl_path
            )
            dict_scores = add_to_dict_scores(dict_scores, sl_tl_name, sl_tl_jsd_score, sl_tl_jaccard_score_result)


            # SL'-TL
            print("\nSL'-TL scores -------------------------------------")
            nmt_sc_train = nmt_sc_config["train_data"]
            nmt_sc_spm = nmt_sc_config["spm"]
            nmt_sc_model_id = nmt_sc_config["sc_model_id"]
            slp_tl_path, slp_tl_name = get_out_f(prime(nmt_sc_config["src"]), nmt_sc_config["tgt"], out_dir, tag="1")
            slp_tl_jsd_score, slp_tl_jaccard_score_result = calc_overlap(
                data1_f=nmt_sc_train,
                spm1=nmt_sc_spm,
                sc_model_id_1=nmt_sc_model_id,
                
                data2_f=nmt_sc_train,
                spm2=nmt_sc_spm,
                sc_model_id_2="DUMMY",
                read_tgt2=True,

                is_parallel=True,
                out_f=slp_tl_path
            )
            dict_scores = add_to_dict_scores(dict_scores, slp_tl_name, slp_tl_jsd_score, slp_tl_jaccard_score_result)

            # pSL-TL between regular en and child language
            print("\npSL-TL scores -------------------------------------")
            pretrain_sc_train = pretrain_sc_config["train_data"] # get "tgt" (regular en instead of en')
            finetune_sc_train = finetune_sc_config["train_data"]
            psl_tl_spm = nmt_spm # using nmt tokenizer that has only seen the source and target language
            psl_tl_path, psl_tl_name = get_out_f(finetune_sc_config["src"], pretrain_sc_config["tgt"], out_dir, tag="1p")
            psl_tl_jsd_score, psl_tl_jaccard_score_result = calc_overlap(
                data1_f=pretrain_sc_train,
                spm1=psl_tl_spm,
                sc_model_id_1=None,
                read_tgt1=True,

                data2_f=finetune_sc_train,
                spm2=psl_tl_spm,
                sc_model_id_2=None,

                is_parallel=False,
                out_f=psl_tl_path
            )
            dict_scores = add_to_dict_scores(dict_scores, psl_tl_name, psl_tl_jsd_score, psl_tl_jaccard_score_result)


            # pSL-TL' between en' and child language
            print("\npSL-TL' scores -------------------------------------")
            pretrain_sc_train = pretrain_sc_config["train_data"]
            finetune_sc_train = finetune_sc_config["train_data"]
            assert pretrain_sc_config["spm"] == finetune_sc_config["spm"]
            psl_tlp_spm = pretrain_sc_config["spm"]
            psl_tlp_path, psl_tlp_name = get_out_f(finetune_sc_config["src"], prime(pretrain_sc_config["tgt"]), out_dir, tag="1p")
            psl_tlp_jsd_score, psl_tlp_jaccard_score_result = calc_overlap(
                data1_f=pretrain_sc_train,
                spm1=psl_tlp_spm,
                sc_model_id_1=pretrain_sc_config["sc_model_id"],

                data2_f=finetune_sc_train,
                spm2=psl_tlp_spm,
                sc_model_id_2=None,

                is_parallel=False,
                out_f=psl_tlp_path
            )
            dict_scores = add_to_dict_scores(dict_scores, psl_tlp_name, psl_tlp_jsd_score, psl_tlp_jaccard_score_result) 
    
    print("DATA COLLECTED:")
    df_scores = pd.DataFrame(dict_scores)
    # df_scores.drop(columns=['Jaccard ↑'], inplace=True)
    print(df_scores)
    write_latex(df_scores, latex_path, column_format="lrr")
    write_long_latex(dict_scores, latex_long_path)

def write_latex(df_scores, latex_path, skip=3, column_format="lr"):
    print("Writing LaTEX")
    latex_str = df_scores.to_latex(index=False, escape=False, column_format=column_format)
    latex_lines = latex_str.splitlines()
    assert latex_lines[3] == "\\midrule"
    l = 6
    line = latex_lines[l]
    while line != "\\bottomrule":
        latex_lines.insert(l, "\\midrule")
        l += skip
        line = latex_lines[l]
    latex_str = "\n".join(latex_lines)
    with open(latex_path, "w") as outf:
        outf.write(latex_str)

def write_long_latex(dict_scores, latex_long_path):
    assert latex_long_path.endswith(".txt")
    latex_long_three_set_path = latex_long_path[:-3] + "three-set.txt"
    latex_long_one_set_path = latex_long_path[:-3] + "one-set.txt"

    dict_len = len(dict_scores[DATA_HEAD])
    for key, values in dict_scores.items():
        assert len(values) == dict_len
    
    lang_pair_scores = {}
    for i in range(dict_len):
        lang_pair = dict_scores[DATA_HEAD][i]
        if lang_pair.startswith("1"):
            tag, lang_pair = lang_pair.split("_")
        else:
            tag = None

        src, tgt = lang_pair.split("-")
        src_is_prime = src.endswith("'")
        tgt_is_prime = tgt.endswith("'")

        if src_is_prime:
            lang_pair_key = f"{src[:-1]}-{tgt}"
        elif tgt_is_prime:
            lang_pair_key = f"{src}-{tgt[:-1]}"
        else:
            lang_pair_key = f"{src}-{tgt}"
        
        if lang_pair_key not in lang_pair_scores:
            lang_pair_scores[lang_pair_key] = {"normal": None, "prime": None, "p-normal": None, "p-tl-prime": None}
        
        score_key = None
        if tag == None:
            assert tgt_is_prime == False
            if src_is_prime:
                score_key = "prime"
            else:
                score_key = "normal"
        else:
            if tgt_is_prime:
                assert src_is_prime == False
                score_key = "p-tl-prime"
                assert tag == "1p"
            elif src_is_prime:
                assert tgt_is_prime == False
                score_key = "prime"
                assert tag == "1"
            else:
                if tag == "1":
                    score_key = "normal"
                else:
                    assert tag == "1p"
                    score_key = "p-normal"
        assert score_key != None

        JSD_score = float(dict_scores[JSD_HEAD][i])

        assert lang_pair_scores[lang_pair_key][score_key] == None
        lang_pair_scores[lang_pair_key][score_key] = JSD_score
    
    for lang_pair, scores in lang_pair_scores.items():
        assert scores["normal"] != None
        assert scores["prime"] != None
        scores_list = [(s, label) for label, s in scores.items() if s is not None]
        max_score, max_label = min(scores_list)
        assert max_score == lang_pair_scores[lang_pair][max_label]
        lang_pair_scores[lang_pair][max_label] = f"\\textbf{{{max_score}}}"
    
    print("LANG PAIR JSD SCORES")
    print(json.dumps(lang_pair_scores, indent=2))

    long_dict = {"Method": ["PL-CL", "PL'-CL"]}
    long_dict.update({lp: [] for lp in lang_pair_scores})
    long_dict_one_set = {"Method": ["SL-TL", "SL'-TL", "p TL-SL", "p TL'-SL"]}
    long_dict_one_set.update({lp: [] for lp in lang_pair_scores})
    for lp, scores in lang_pair_scores.items():
        src, tgt = lp.split("-")
        if scores["p-tl-prime"] == None:
            assert scores["p-normal"] == None
            assert all(
                [scores["normal"] is not None, 
                 scores["prime"] is not None])
            long_dict[lp] += [scores["normal"], scores["prime"]]
        else:
            assert all(
                [scores["normal"] is not None, 
                 scores["prime"] is not None, 
                 scores["p-normal"] is not None, 
                 scores["p-tl-prime"] is not None])
            long_dict_one_set[lp] += [scores["normal"], scores["prime"], scores["p-normal"], scores["p-tl-prime"]]
    
    long_dict, long_dict_is_empty = remove_empty_columns(long_dict)
    long_dict_one_set, long_dict_one_set_is_empty = remove_empty_columns(long_dict_one_set)

    long_df = pd.DataFrame(long_dict)
    long_df_one_set = pd.DataFrame(long_dict_one_set)

    column_format = "l" + "".join(["r" for lp in lang_pair_scores])
    if not long_dict_is_empty:
        write_latex(long_df, latex_long_three_set_path, column_format=column_format)
    if not long_dict_one_set_is_empty:
        write_latex(long_df_one_set, latex_long_one_set_path, skip=4, column_format=column_format)

def remove_empty_columns(long_dict):
    dict_len = len(long_dict["Method"])
    to_pop = []
    for key, values in long_dict.items():
        assert len(values) in [0, dict_len]
        if len(values) == 0:
            to_pop.append(key)
    for key in to_pop:
        long_dict.pop(key)
    return long_dict, list(long_dict.keys()) == ["Method"]

def add_to_dict_scores(dict_scores, name, jsd, jaccard):
    assert sorted(list(dict_scores.keys())) == sorted([DATA_HEAD, JSD_HEAD, JACCARD_HEAD])
    assert len(dict_scores[DATA_HEAD]) == len(dict_scores[JSD_HEAD]) == len(dict_scores[JACCARD_HEAD])
    dict_scores[DATA_HEAD].append(name)
    dict_scores[JSD_HEAD].append(jsd)
    dict_scores[JACCARD_HEAD].append(jaccard["global"])
    assert len(dict_scores[DATA_HEAD]) == len(dict_scores[JSD_HEAD]) == len(dict_scores[JACCARD_HEAD])
    return dict_scores

def get_out_f(lang1, lang2, out_dir, tag=""):
    if tag:
        tag += "_"
    f_name = f"{tag}{lang1}-{lang2}"
    f_path = os.path.join(out_dir, f_name + ".json")
    f_name = "$" + f_name + "$"
    return f_path, f_name

def prime(lang):
    return f"{lang}'"

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
    assert len(matches) in [0, 1], f"Found more than one file matching r'{regex}' in {d_path}: {matches}"
    if len(matches) == 1:
        return matches[0]
    else:
        print(f"Found no matches for r'{regex}' in {d_path}")
        return None

def read_yaml(f):
    if f == None:
        return None
    print(f"reading yaml: `{f}`")
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configs_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("-o", "--out_dir", default="/home/hatch5o6/nobackup/archive/CognateMT/vocab_overlap")
    parser.add_argument("-l", "--lang_pairs", default=None, help="comma-delimited")
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
    lang_pairs = args.lang_pairs
    if lang_pairs:
        lang_pairs = [l.strip() for l in lang_pairs.split(",")]
    main(args.configs_dir, args.out_dir, lang_pairs)
