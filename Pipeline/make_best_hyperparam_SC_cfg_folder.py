import argparse
import os
import shutil
import pandas as pd


def main(
    best_configs,
    hyperparam_search_cfgs,
    smt_cfgs,
    out_dir
):
    if os.path.exists(out_dir):
        print(f"{out_dir} exists. Do you want to delete it?")
        answer = None
        while answer not in ["Y", "N"]:
            answer = input("\t(Y or N):").upper()
        assert answer in ["Y", "N"]
        if answer == "Y":
            print(f"Destroying {out_dir}")
            shutil.rmtree(out_dir)
        elif answer == "N":
            print("Okay. Will now end.")
            exit()
    assert not os.path.exists(out_dir)
    print("Creating", out_dir)
    os.mkdir(out_dir)

    best_models = {}
    best_configs_df = pd.read_excel(best_configs)
    for idx, row in best_configs_df.iterrows():
        lang = row["LANG"]
        criteria = row["CRITERIA"]
        assert criteria in ["BLEU", "chrF"]
        model_id = row["RNN_ID"]
        model_type = row["model_type"]
        if lang not in best_models:
            best_models[lang] = {"BLEU": None, "chrF": None}
        
        assert best_models[lang][criteria] == None
        best_models[lang][criteria] = (model_type, model_id)
    
    CFG_FILES = set()
    for lang in best_models:
        for criteria in best_models[lang]:
            model_type, model_id = best_models[lang][criteria]
            assert model_type in ["SMT", "bigru"]
            if model_type == "bigru":
                model_type = "RNN"
                cfg_file = os.path.join(hyperparam_search_cfgs, f"{lang}.{model_id}.cfg")
                if not os.path.exists(cfg_file):
                    cfg_file = os.path.join(hyperparam_search_cfgs, f"{lang}.ATT.{model_id}.cfg")
            elif model_type == "SMT":
                assert model_id == 288
                cfg_file = os.path.join(smt_cfgs, f"{lang}.smt.cfg")
                if not os.path.exists(cfg_file):
                    cfg_file = os.path.join(smt_cfgs, f"{lang}.ATT.smt.cfg")
            
            CFG_FILES.add(cfg_file)

    for cfg_file in sorted(list(CFG_FILES)):
        assert os.path.exists(cfg_file)
        cfg_name = cfg_file.split("/")[-1]
        new_cfg_file = os.path.join(out_dir, cfg_name)
        if os.path.exists(new_cfg_file):
            print(f"{new_cfg_file} already exists!!")
        assert not os.path.exists(new_cfg_file)
        shutil.copyfile(cfg_file, new_cfg_file)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_configs", help="path to best_configs.xlsx resulting from hyperparameter search")
    parser.add_argument("--hyperparam_search_SC_cfgs", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH")
    parser.add_argument("--smt_SC_cfgs", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT")
    parser.add_argument("--out_dir", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC-BEST")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("######################################")
    print("# make_best_hyperparam_SC_cfg_folder #")
    print("######################################")
    args = get_args()
    main(
        best_configs=args.best_configs,
        hyperparam_search_cfgs=args.hyperparam_search_SC_cfgs,
        smt_cfgs=args.smt_SC_cfgs,
        out_dir=args.out_dir
    )
