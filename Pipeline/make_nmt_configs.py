import argparse
import os
import shutil
import yaml
import copy
from tqdm import tqdm

TEST_FOLDER = "/home/hatch5o6/Cognate/code/NMT/configs/TEST"

def make_configs(
    cfgs,
    out_dir,
    template_f,
    nmt_data,
    nmt_aug_data,
    SAVE_PARENT_DIR,
    TOKENIZERS_DIR,
    TESTING=False,
    NO_V=None
):
    if os.path.exists(out_dir):
        print("DELETING", out_dir)
        shutil.rmtree(out_dir)
    print("CREATING", out_dir)
    os.mkdir(out_dir)

    config_template = read_yaml(template_f)
    PATHS = {}

    print("Making configs")
    for cfg in tqdm(os.listdir(cfgs)):
        if TESTING and cfg not in ["bho-hi.cfg", "bn-as.cfg"]: continue
        if cfg in ["en-djk.ATT.cfg"]: continue
        print("CONFIG:", cfg)
        cfg_path = os.path.join(cfgs, cfg)
        NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, COG_SRC, COG_TGT = read_cfg(cfg_path)
        print("\tNMT_SRC:", NMT_SRC)
        print("\tNMT_TGT:", NMT_TGT)
        print("\tAUG_SRC:", AUG_SRC)
        print("\tAUG_TGT:", AUG_TGT)
        print("\tCOG_SRC:", COG_SRC)
        print("\tCOG_TGT:", COG_TGT)
        
        lang_pair = f"{NMT_SRC}-{NMT_TGT}"
        lang_out_dir = os.path.join(out_dir, lang_pair)
        if os.path.exists(lang_out_dir):
            print("deleting", lang_out_dir)
            shutil.rmtree(lang_out_dir)
        print("creating", lang_out_dir)
        os.mkdir(lang_out_dir)

        assert NMT_TGT == AUG_TGT
        assert AUG_SRC == COG_SRC

        SHOULD_AUGMENT = should_augment(NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, COG_SRC, COG_TGT)
        print("SHOULD_AUGMENT:", SHOULD_AUGMENT)

        if lang_pair not in PATHS:
            PATHS[lang_pair] = []

        no_v_tag = ""
        if NO_V != None:
            no_v_tag = f".no_overlap_{NO_V}"

        if SHOULD_AUGMENT:
            assert NMT_SRC == COG_TGT
            # AUGMENTED
            # plain
            aug_config = copy.deepcopy(config_template)
            aug_config["src"] = NMT_SRC
            aug_config["tgt"] = NMT_TGT
            aug_config["train_data"] = os.path.join(nmt_aug_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}", f"train{no_v_tag}.csv")
            aug_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            aug_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
            # aug_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"{AUG_SRC}.{NMT_SRC}/{AUG_SRC}.{NMT_SRC}")
            # aug_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            aug_config["spm"] = os.path.join(TOKENIZERS_DIR, f"{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}")
            aug_config["upsample"] = True
            aug_config["save"] = os.path.join(SAVE_PARENT_DIR, f"AUGMENT.{NMT_SRC}-{NMT_TGT}")
            PATHS[lang_pair].append(write_yaml(aug_config, os.path.join(lang_out_dir, f"AUGMENT.{NMT_SRC}-{NMT_TGT}.yaml")))

            # sc
            aug_sc_config = copy.deepcopy(config_template)
            aug_sc_config["src"] = NMT_SRC
            aug_sc_config["tgt"] = NMT_TGT                             # COG_TGT should match NMT_SRC, if we're making this
            aug_sc_config["train_data"] = os.path.join(nmt_aug_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}", f"train{no_v_tag}.csv")
            aug_sc_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            aug_sc_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
            # aug_sc_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}/SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}")
            # aug_sc_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            aug_sc_config["spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}")
            aug_sc_config["upsample"] = True
            aug_sc_config["save"] = os.path.join(SAVE_PARENT_DIR, f"AUGMENT.SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}")
            PATHS[lang_pair].append(write_yaml(aug_sc_config, os.path.join(lang_out_dir, f"AUGMENT.SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}.yaml")))

            # PRETRAIN
            # plain
            pretrain_config = copy.deepcopy(config_template)
            pretrain_config["src"] = AUG_SRC
            pretrain_config["tgt"] = AUG_TGT
            pretrain_config["train_data"] = os.path.join(nmt_data, "PLAIN", f"{AUG_SRC}-{AUG_TGT}", f"train{no_v_tag}.csv")
            pretrain_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{AUG_SRC}-{AUG_TGT}_dev_test", f"val{no_v_tag}.csv")
            pretrain_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{AUG_SRC}-{AUG_TGT}_dev_test", "test.csv")
            # pretrain_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"{AUG_SRC}.{NMT_SRC}/{AUG_SRC}.{NMT_SRC}")
            # pretrain_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            pretrain_config["spm"] = os.path.join(TOKENIZERS_DIR, f"{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}")
            pretrain_config["upsample"] = False
            pretrain_config["save"] = os.path.join(SAVE_PARENT_DIR, f"PRETRAIN.{AUG_SRC}-{AUG_TGT}")
            PATHS[lang_pair].append(write_yaml(pretrain_config, os.path.join(lang_out_dir, f"PRETRAIN.{AUG_SRC}-{AUG_TGT}.yaml")))
            
            # sc
            pretrain_sc_config = copy.deepcopy(config_template)
            pretrain_sc_config["src"] = AUG_SRC
            pretrain_sc_config["tgt"] = AUG_TGT                                    #COG_SRC==AUG_SRC
            pretrain_sc_config["train_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}", f"train{no_v_tag}.csv")
            pretrain_sc_config["val_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}_dev_test", f"val{no_v_tag}.csv")
            pretrain_sc_config["test_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}_dev_test", "test.csv")
            # pretrain_sc_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}/SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}")
            # pretrain_sc_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            pretrain_sc_config["spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}")
            pretrain_sc_config["upsample"] = False
            pretrain_sc_config["save"] = os.path.join(SAVE_PARENT_DIR, f"PRETRAIN.SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}")
            PATHS[lang_pair].append(write_yaml(pretrain_sc_config, os.path.join(lang_out_dir, f"PRETRAIN.SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}.yaml")))

            # FINETUNE
            # from pretrain plain
            finetune_config = copy.deepcopy(config_template)
            finetune_config["src"] = NMT_SRC
            finetune_config["tgt"] = NMT_TGT
            finetune_config["train_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}", f"train{no_v_tag}.csv")
            finetune_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            finetune_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
            # finetune_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"{AUG_SRC}.{NMT_SRC}/{AUG_SRC}.{NMT_SRC}")
            # finetune_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            finetune_config["spm"] = os.path.join(TOKENIZERS_DIR, f"{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}/{COG_SRC}-{COG_TGT}_{NMT_TGT}")
            finetune_config["upsample"] = False
            finetune_config["save"] = os.path.join(SAVE_PARENT_DIR, f"FINETUNE.{AUG_SRC}-{AUG_TGT}>>{NMT_SRC}-{NMT_TGT}")
            finetune_config["from_pretrained"] = pretrain_config["save"] + f"_TRIAL_s={pretrain_config['seed']}" 
            PATHS[lang_pair].append(write_yaml(finetune_config, os.path.join(lang_out_dir, f"FINETUNE.{AUG_SRC}-{AUG_TGT}>>{NMT_SRC}-{NMT_TGT}.yaml")))

            # from pretrain sc
            finetune_sc_config = copy.deepcopy(config_template)
            finetune_sc_config["src"] = NMT_SRC
            finetune_sc_config["tgt"] = NMT_TGT
            finetune_sc_config["train_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}", f"train{no_v_tag}.csv")
            finetune_sc_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            finetune_sc_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
            # finetune_sc_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}/SC_{COG_SRC}2{COG_TGT}.{NMT_SRC}")
            # finetune_sc_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            finetune_sc_config["spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_{NMT_TGT}")
            finetune_sc_config["upsample"] = False
            finetune_sc_config["save"] = os.path.join(SAVE_PARENT_DIR, f"FINETUNE.SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}>>{NMT_SRC}-{NMT_TGT}")
            finetune_sc_config["from_pretrained"] = pretrain_sc_config["save"] + f"_TRIAL_s={pretrain_sc_config['seed']}" 
            PATHS[lang_pair].append(write_yaml(finetune_sc_config, os.path.join(lang_out_dir, f"FINETUNE.SC_{COG_SRC}2{COG_TGT}-{AUG_TGT}>>{NMT_SRC}-{NMT_TGT}.yaml")))
        else:
            # Not augmenting
            assert NMT_SRC == AUG_SRC == COG_SRC
            assert NMT_TGT == AUG_TGT == COG_TGT

            # plain
            nmt_config = copy.deepcopy(config_template)
            nmt_config["src"] = NMT_SRC
            nmt_config["tgt"] = NMT_TGT
            nmt_config["train_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}", f"train{no_v_tag}.csv")
            nmt_config["val_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            nmt_config["test_data"] = os.path.join(nmt_data, "PLAIN", f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
            # nmt_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_SRC}/{NMT_SRC}")
            # nmt_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            nmt_config["spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_SRC}_{NMT_TGT}/{NMT_SRC}_{NMT_TGT}/{NMT_SRC}_{NMT_TGT}")
            nmt_config["upsample"] = False
            nmt_config["save"] = os.path.join(SAVE_PARENT_DIR, f"NMT.{NMT_SRC}-{NMT_TGT}")
            PATHS[lang_pair].append(write_yaml(nmt_config, os.path.join(lang_out_dir, f"NMT.{NMT_SRC}-{NMT_TGT}.yaml")))

            # sc
            nmt_sc_config = copy.deepcopy(config_template)
            nmt_sc_config["src"] = NMT_SRC
            nmt_sc_config["tgt"] = NMT_TGT
            nmt_sc_config["train_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}", f"train{no_v_tag}.csv")
            nmt_sc_config["val_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}_dev_test", f"val{no_v_tag}.csv")
            nmt_sc_config["test_data"] = os.path.join(nmt_data, "SC", f"SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}_dev_test", "test.csv")
            # nmt_sc_config["src_spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}/SC_{COG_SRC}2{COG_TGT}")
            # nmt_sc_config["tgt_spm"] = os.path.join(TOKENIZERS_DIR, f"{NMT_TGT}/{NMT_TGT}")
            nmt_sc_config["spm"] = os.path.join(TOKENIZERS_DIR, f"SC_{COG_SRC}2{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}_{NMT_TGT}/SC_{COG_SRC}2{COG_TGT}_{NMT_TGT}")
            nmt_sc_config["upsample"] = False
            nmt_sc_config["save"] = os.path.join(SAVE_PARENT_DIR, f"NMT.SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}")
            PATHS[lang_pair].append(write_yaml(nmt_sc_config, os.path.join(lang_out_dir, f"NMT.SC_{COG_SRC}2{COG_TGT}-{NMT_TGT}.yaml")))


    # if TESTING:
    TEST(PATHS, config_template)

def TEST(paths, config_template):
    SKIP_KEYS = ["src", "tgt", "train_data", "val_data", "test_data", 
                #  "src_spm", "tgt_spm",
                 "spm",
                "upsample", "save", "from_pretrained"]
    total = 0
    passed = 0
    for d in os.listdir(TEST_FOLDER):
        d_path = os.path.join(TEST_FOLDER, d)
        test_paths = paths[d]
        for test_f in test_paths:
            test_f_name = test_f.split("/")[-1]
            gt_path = os.path.join(d_path, test_f_name)
            
            if os.path.exists(gt_path):
                total += 1
                print("------------------------------------")
                print("comparing GT", gt_path)
                print("\tto", test_f)
                gt_content = read_file(gt_path)
                test_content = read_file(test_f)
                if gt_content != test_content:
                    print(f"!!!!! GT:", gt_path)
                    print("\t DOES NOT EQUAL OUTPUT", test_f)
                else:
                    print(":) GT:", gt_path)
                    print("\t EQUALS OUTPUT", test_f)

                    passed += 1

            if os.path.exists(gt_path):
                gt_yaml = read_yaml(gt_path)
            else:
                gt_yaml = None

            test_yaml = read_yaml(test_f)
            for key, v in config_template.items():
                if key in SKIP_KEYS:
                    continue
                # print("---------")
                # print("key", key)
                # print("v", type(v), v)
                # print("gt_yaml", key, type(gt_yaml[key]), gt_yaml[key])
                # print("test_yaml", key, type(test_yaml[key]), test_yaml[key])
                if gt_yaml:
                    assert gt_yaml[key] == v
                assert test_yaml[key] == v
            if gt_yaml:
                assert "from_pretrained" in gt_yaml
            assert "from_pretrained" in test_yaml
            if "FINETUNE." in gt_path:
                assert "FINETUNE." in test_f
                if gt_yaml:
                    assert gt_yaml["from_pretrained"] is not None
                assert test_yaml["from_pretrained"] is not None
            else:
                assert "FINETUNE." not in test_f
                if gt_yaml:
                    assert gt_yaml["from_pretrained"] is None
                assert test_yaml["from_pretrained"] is None

    print("================================")
    print("Asserted default values for standard keys")
    print(f"{passed} / {total} FILES PASSED")

def read_file(f):
    with open(f) as inf:
        content = inf.read()
    return content.strip() + "\n"

def should_augment(NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, COG_SRC, COG_TGT):
    if NMT_SRC == AUG_SRC == COG_SRC and NMT_TGT == AUG_TGT == COG_TGT:
        return False
    else:
        assert NMT_TGT == AUG_TGT
        assert AUG_SRC == COG_SRC
        assert NMT_SRC == COG_TGT

        return True

def read_yaml(f):
    with open(f, 'r') as inf:
        data = yaml.safe_load(inf)
    return data

def write_yaml(data, f):
    with open(f, "w") as outf:
        for key, v in data.items():
            if key == "src":
                outf.write("# outputs\n")
            elif key == "from_pretrained":
                outf.write("# finetune?\n")
            elif key == "train_data":
                outf.write("# data\n")
            # elif key == "src_spm":
            elif key == "spm":
                outf.write("# tokenizers\n")
            elif key == "n_gpus":
                outf.write("# training\n")
            elif key == "encoder_layers":
                outf.write("# config\n")

            if v is None:
                v = "null"
            outf.write(f"{key}: {v}\n")

            if key in ["little_verbose", "from_pretrained", "upsample", 
                    #    "tgt_spm", 
                       "spm",
                       "qos", "test_batch_size", "val_interval", "learning_rate", "device", "encoder_layerdrop", "decoder_layerdrop"]:
                outf.write("\n")
            
        # yaml.dump(data, outf, sort_keys=False)
    return f

def read_cfg(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    NMT_SRC=None
    NMT_TGT=None
    AUG_SRC=None
    AUG_TGT=None
    SRC=None
    TGT=None
    for line in lines:
        if line.startswith("NMT_SRC="):
            assert NMT_SRC is None
            NMT_SRC = line.split("NMT_SRC=")[-1]
        elif line.startswith("NMT_TGT="):
            assert NMT_TGT is None
            NMT_TGT = line.split("NMT_TGT=")[-1]
        elif line.startswith("AUG_SRC="):
            assert AUG_SRC is None
            AUG_SRC = line.split("AUG_SRC=")[-1]
        elif line.startswith("AUG_TGT="):
            assert AUG_TGT is None
            AUG_TGT = line.split("AUG_TGT=")[-1]
        elif line.startswith("SRC="):
            assert SRC is None
            SRC = line.split("SRC=")[-1]
        elif line.startswith("TGT="):
            assert TGT is None
            TGT = line.split("TGT=")[-1]
    assert all([NMT_SRC is not None, NMT_TGT is not None, AUG_SRC is not None, AUG_TGT is not None, SRC is not None, TGT is not None])
    return NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, SRC, TGT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgs", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC")
    parser.add_argument("--out_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("--template", default="/home/hatch5o6/Cognate/code/NMT/configs/template.yaml")
    parser.add_argument("--nmt_data", default="/home/hatch5o6/Cognate/code/NMT/data")
    parser.add_argument("--nmt_augmented_data", default="/home/hatch5o6/Cognate/code/NMT/augmented_data")
    parser.add_argument("--SAVE_PARENT_DIR", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates")
    parser.add_argument("--TOKENIZERS_DIR", default="/home/hatch5o6/nobackup/archive/CognateMT/spm_models")
    parser.add_argument("--TESTING", action="store_true", help="if passed, will only create bho-hi and bn-as files")
    parser.add_argument("--NO_V", help="no overlap version, e.g. v1 will add file tag '.no_overlap_v1.csv' on train and val .csvs", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    make_configs(
        cfgs=args.cfgs,
        out_dir=args.out_dir,
        template_f=args.template,
        nmt_data=args.nmt_data,
        nmt_aug_data=args.nmt_augmented_data,
        SAVE_PARENT_DIR=args.SAVE_PARENT_DIR,
        TOKENIZERS_DIR=args.TOKENIZERS_DIR,
        TESTING=args.TESTING,
        NO_V=args.NO_V
    )