import argparse
import pandas as pd
import os
import shutil
import time


def main(
    best_configs,
    CopperMT_dir,
    COGNATE_TRAIN_dir,
    MOVE_TO_dir,
    CLEAR_MOVE_TO=False,
    BACKUP_TO=None
):
    assert BACKUP_TO != None

    print(f"Will clean up based on results from: {best_configs}")
    print("Are you sure you want to continue?")
    answer = None
    while answer not in ["Y", "N"]:
        answer = input("\t(Y or N): ").upper()
    assert answer in ["Y", "N"]
    if answer == "N":
        print("Okay. Will end without making any changes.")
        exit()
    elif answer == "Y":
        print("Here we go, baby. You're locked in now! Mwahahahaha!")
        time.sleep(3)

    best_configs_name = best_configs.split("/")[-2]
    BACKUP_TO = os.path.join(BACKUP_TO, best_configs_name)
    if os.path.exists(BACKUP_TO):
        print("Do you want to delete backup?", BACKUP_TO)
        answer = None
        while answer not in ["Y", "N"]:
            answer = input("\t(Y or N): ").upper()
        assert answer in ["Y", "N"]
        if answer == "Y":
            print("Deleting", BACKUP_TO)
            shutil.rmtree(BACKUP_TO)
        else:
            print("Okay. Will terminate program.")
            exit()

    BACKUP_COGNATE_TRAIN = os.path.join(BACKUP_TO, "COGNATE_TRAIN")
    BACKUP_CopperMT = os.path.join(BACKUP_TO, "CopperMT")
    if not os.path.exists(BACKUP_TO):
        print("Making BACKUP folder:", BACKUP_TO)
        os.mkdir(BACKUP_TO)
        os.mkdir(BACKUP_COGNATE_TRAIN)
        os.mkdir(BACKUP_CopperMT)
    assert os.path.exists(BACKUP_COGNATE_TRAIN)
    assert os.listdir(BACKUP_COGNATE_TRAIN) == []
    assert os.path.exists(BACKUP_CopperMT)
    assert os.listdir(BACKUP_CopperMT) == []
    

    best_configs_df = pd.read_excel(best_configs)
    best_models = {}
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



    # Clean up COGNATE_TRAIN
    print("CLEANING COGNATE_TRAIN")
    KEEP_DIRS = set()
    for lang in best_models:
        for d in os.listdir(COGNATE_TRAIN_dir):
            if d == "_OLD":
                KEEP_DIRS.add(d)
                continue
            d_lang, d_model_type_and_id, d_seed = tuple(d.split("_"))
            if d_lang.endswith(".ATT"):
                d_lang = d_lang[:-4]
            if d_lang != lang: continue

            d_model_type, d_id = tuple(d_model_type_and_id.split("-"))
            if d_id == "null":
                d_id = 288
            for crit, (crit_model_type, crit_id) in best_models[lang].items():
                if crit_model_type == "bigru":
                    crit_model_type = "RNN"
                if d_lang == lang and d_model_type == crit_model_type and int(d_id) == int(crit_id):
                    dpath_x = os.path.join(COGNATE_TRAIN_dir, d)
                    if dpath_x not in KEEP_DIRS:
                        print("\tKEEPING", dpath_x)
                        KEEP_DIRS.add(dpath_x)
                        shutil.copytree(dpath_x, os.path.join(BACKUP_COGNATE_TRAIN, d))
    for d in os.listdir(COGNATE_TRAIN_dir):
        d_path = os.path.join(COGNATE_TRAIN_dir, d)
        if d_path not in KEEP_DIRS:
            pass
            shutil.rmtree(d_path) # uncomment after testing


    # Clean up CopperMT
    MOVE_TO_name = MOVE_TO_dir.split("/")[-1]

    if not os.path.exists(MOVE_TO_dir):
        os.mkdir(MOVE_TO_dir)
    MOVE_TO_dir = os.path.join(MOVE_TO_dir, best_configs_name)
    if CLEAR_MOVE_TO == True:
        if os.path.exists(MOVE_TO_dir):
            print("Clearing MOVE_TO_dir:", MOVE_TO_dir)
            print(f"Are you sure you want to clear away: `{MOVE_TO_dir}`?")
            answer = None
            while answer not in ["Y", "N"]:
                answer = input("\t(Y or N): ").upper()
            assert answer in ["Y", "N"]
            if answer == "Y":
                print("Okay. Removing", MOVE_TO_dir)
                shutil.rmtree(MOVE_TO_dir)
                assert not os.path.exists(MOVE_TO_dir)
                print("It's gone now.")
    if not os.path.exists(MOVE_TO_dir):
        print("Making MOVE_TO_dir:", MOVE_TO_dir)
        os.mkdir(MOVE_TO_dir)
    
    print("CLEANING CopperMT")
    COPPER_KEEP_DIRS = set()
    for lang in best_models:
        for copper_dir in os.listdir(CopperMT_dir):
            if copper_dir in ["_OLD", MOVE_TO_name]:
                COPPER_KEEP_DIRS.add(copper_dir)
                continue

            c_src, c_tgt, c_model_type_and_id, c_seed = tuple(copper_dir.split("_"))
            lang_pair = f"{c_src}-{c_tgt}"
            seed = int(c_seed.split("-")[1])
            c_model_type, c_id = tuple(c_model_type_and_id.split("-"))
            if c_id == "null":
                c_id = 288
            if lang_pair != lang: continue
            
            for crit, (crit_model_type, crit_id) in best_models[lang_pair].items():
                if crit_model_type == "bigru":
                    crit_model_type = "RNN"
                if lang_pair == lang and c_model_type == crit_model_type and int(c_id) == int(crit_id):
                    copper_dir_path_x = os.path.join(CopperMT_dir, copper_dir)
                    if copper_dir_path_x not in COPPER_KEEP_DIRS:
                        COPPER_KEEP_DIRS.add(copper_dir_path_x)
                        print("\tKEEPING", copper_dir_path_x)
                        shutil.copytree(copper_dir_path_x, os.path.join(BACKUP_CopperMT, copper_dir))
    
    print("COPPER KEEP DIRS")
    for x in COPPER_KEEP_DIRS:
        print(x)
    # exit()

    for copper_dir in os.listdir(CopperMT_dir):
        if copper_dir in ["_OLD", MOVE_TO_name]: continue
        copper_dir_path = os.path.join(CopperMT_dir, copper_dir)
        if copper_dir_path not in COPPER_KEEP_DIRS:
            c_src, c_tgt, c_model_type_and_id, c_seed = tuple(copper_dir.split("_"))
            seed = int(c_seed.split("-")[1])
            c_model_type, c_id = tuple(c_model_type_and_id.split("-"))

            # print("copper_dir_path: ", copper_dir_path)

            if c_model_type == "RNN":
                checkpoint_dir_path = os.path.join(CopperMT_dir, copper_dir, f"workspace/reference_models/bilingual/rnn_{c_src}-{c_tgt}/{seed}/checkpoints")
                # print("\tchkpt:", checkpoint_dir_path)
                if os.path.exists(checkpoint_dir_path):
                    # print("REMOVE CHECKPOINT", checkpoint_dir_path)
                    shutil.rmtree(checkpoint_dir_path)
                    pass
            else:
                assert c_model_type == "SMT"
        else:
            print(f"KEEPING CHECKPOINT IN {copper_dir}")
        
        if copper_dir not in ["_OLD", MOVE_TO_name]:
            if copper_dir_path not in COPPER_KEEP_DIRS:
                pass
                # print("MOVING", copper_dir_path, "TO", MOVE_TO_dir)
                shutil.move(copper_dir_path, MOVE_TO_dir) # uncomment after testing
            else:
                print("NOT MOVING", copper_dir_path)
    
        



        


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_configs", help="path to best_configs.xlsx resulting from hyperparameter search")
    parser.add_argument("--CopperMT", default="/home/hatch5o6/nobackup/archive/CopperMT")
    parser.add_argument("--COGNATE_TRAIN", default="/home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN")
    parser.add_argument("--MOVE_TO", help="Will move CopperMT contents (except the model checkpoint) to this directory for records", default="/home/hatch5o6/nobackup/archive/CopperMT/_hyperparam_results_leftovers")
    parser.add_argument("--CLEAR_MOVE_TO", action="store_true", default=False)
    parser.add_argument("--BACKUP_TO", default="/home/hatch5o6/nobackup/archive/CopperMT_BACKUP")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("#################################")
    print("# clean_up_hyperparam_search.py #")
    print("#################################")
    args = get_args()
    main(
        best_configs=args.best_configs,
        CopperMT_dir=args.CopperMT,
        COGNATE_TRAIN_dir=args.COGNATE_TRAIN,
        MOVE_TO_dir=args.MOVE_TO,
        CLEAR_MOVE_TO=args.CLEAR_MOVE_TO,
        BACKUP_TO=args.BACKUP_TO
    )