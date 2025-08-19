import argparse
import os
import re

COPPER_MT_DIR = "/home/hatch5o6/nobackup/archive/CopperMT"

hyperparams = {
    "attention": ["attention_type"],

    "enc_layer": ["encoder_layers"],
    "enc_emb_dim": ["encoder_embed_dim"],
    "enc_hid_dim": ["encoder_hidden_size"],

    "dec_layer": ["decoder_layers"],
    "dec_emb_dim": ["decoder_embed_dim"],
    "dec_hid_dim": ["decoder_hidden_size"],

    "batch_size": ["batch_size", "batch_size_valid"],
    "dropout": ["dropout"],
    "learning_rate": ["lr"],
}

def check_hyp_params(
    dir_template,
    cog_lang_pair,
    seed,
    rnn_hyp_params_dir,
    slurm_outputs_dir
):
    slurm_outputs_fs = get_slurm_outputs(slurm_outputs_dir)
    TOTAL = 0
    FILE_PARAMS_MATCH_MODEL = 0
    for f in os.listdir(rnn_hyp_params_dir):
        if f == "manifest.json": continue
        f_path = os.path.join(rnn_hyp_params_dir, f)
        if os.path.isdir(f_path): continue

        TOTAL += 1
        PARAMS_MATCH = True

        f_split = f.split(".")
        assert len(f_split) == 3
        assert f_split[1] == "rnn"
        assert f_split[2] == "txt"
        rnn_id = int(f_split[0])

        print("----------------------------------------------------")
        print("RNN HYPS:", f_path)
        print("RNN_ID:", rnn_id)

        COPPER_MT_cog_lang_pair = cog_lang_pair.replace("-", "_")
        dir_name = dir_template.replace("{COG_LANG_PAIR}", COPPER_MT_cog_lang_pair).replace("{RNN_HYPERPARAMS_ID}", str(rnn_id)).replace("{SEED}", str(seed))
        dir_path = os.path.join(COPPER_MT_DIR, dir_name)
        print("Examining", dir_path)
        assert os.path.exists(dir_path)
        
        dir_rnn_params_f = os.path.join(dir_path, f"inputs/parameters/bilingual_default/default_parameters_rnn_{cog_lang_pair}.txt")
        print(f"COPPERMT RNN HYPS:", dir_rnn_params_f)

        print("asserting rnn hyperparams file matches that in COPPER MT cog lang dir")
        assert files_are_equal(f_path, dir_rnn_params_f)

        params = read_rnn_params(dir_rnn_params_f)

        print("Getting slurm outputs:", (cog_lang_pair, str(rnn_id)))
        slurm_output_f = slurm_outputs_fs[(cog_lang_pair, str(rnn_id))]
        model_hyp_params = get_model_hyper_params_from_slurm_output(slurm_output_f)


        for p, v in params.items():
            print(f"RNN FILE PARAM {p} = {v}")
            if p == "model_type":
                assert v == "bigru"
            elif p in ["share_encoder", "share_decoder"]:
                assert v == ""
            else:
                model_ps = hyperparams[p]
                print("RNN FILE PARAM -> ACTUAL MODEL PARAM")
                print(f"{p} -> {model_ps}")
                for model_p in model_ps:
                    model_p_val = model_hyp_params[model_p]

                    if model_p == "lr":
                        assert model_p_val.startswith("[")
                        assert model_p_val.endswith("]")
                        # print("MODEL LEARNING RATE:", model_p_val)
                        model_p_val = model_p_val[1:-1]
                        # print("! FIXED MODEL LEARNING RATE: ", model_p_val)
                    else:
                        if model_p_val.endswith("'"):
                            assert model_p_val.startswith("'")
                        if model_p_val.startswith("'"):
                            assert model_p_val.endswith("'")
                            model_p_val = model_p_val[1:-1]
                    
                    print(f"\t(value) {v} -> {model_p_val}")
                    if v != model_p_val:
                        print(f"\tRNN file param is {v}")
                        print(f"\tBUT actual model param is {model_p_val}")
                        PARAMS_MATCH = False
        if PARAMS_MATCH:
            print("\nFILE PARAMS MATCH :)")
            FILE_PARAMS_MATCH_MODEL += 1
    
    print("\n\nMATCHES:", FILE_PARAMS_MATCH_MODEL)
    print("TOTAL:", TOTAL)
        
                
def files_are_equal(f1, f2):
    print("\tComparing RNN PARAMS FILES:")
    print(f"\t\t-{f1}")
    print(f"\t\t-{f2}")

    with open(f1) as f1inf:
        f1_content = f1inf.read()

    with open(f2) as f2inf:
        f2_content = f2inf.read()
    
    print("\t\tMatch?", f1_content == f2_content)

    return f1_content == f2_content

def get_slurm_outputs(d):
    slurm_output_fs = {}
    for f in os.listdir(d):
        prefix, lang_pair, rnn_id, cfg, out = tuple(f.split("."))
        assert prefix.endswith("_hyper_param_search")
        assert cfg == "cfg"
        assert out == "out"

        f_path = os.path.join(d, f)
        assert (lang_pair, rnn_id) not in slurm_output_fs
        slurm_output_fs[(lang_pair, rnn_id)] = f_path
    return slurm_output_fs

def get_model_hyper_params_from_slurm_output(f):
    print("Getting actual model hyper params from:")
    print("\t", f)
    features = {}
    with open(f) as inf:
        line = inf.readline()
        while True:
            line = line.strip()
            if line == "-- Training SC MODEL --":
                break
            line = inf.readline()

        assert line == "-- Training SC MODEL --"
        line = inf.readline()
        assert line == "    TYPE=RNN\n"
        line = inf.readline()
        assert line.startswith("    bash ")
        line = inf.readline().strip()
        assert line == "########## main_nmt_bilingual_full_brendan.sh ##########"

        for i in range(23):
            line = inf.readline()
        line = line.strip()

        datetime, INFO, fairseq_cli, namespace = [l.strip() for l in line.split("|")]
        assert INFO == "INFO"
        assert fairseq_cli == "fairseq_cli.train"
        assert namespace.startswith("Namespace(")
        assert namespace.endswith(")")
        namespace = namespace[10:-1]

        for model_ps in hyperparams.values():
            for model_p in model_ps:
                print(model_p)
                q_match = re.findall(f"(?<=\s){model_p}=\'.*?\'(?=\,)", namespace)
                print("\tq_match:", q_match)
                normal_match = re.findall(f"(?<=\s){model_p}=[^\'\,\s]+(?=\,)", namespace)
                print("\tnormal_match:", normal_match)
                if len(q_match) > 0:
                    assert len(normal_match) == 0
                if len(normal_match) > 0:
                    assert len(q_match) == 0
                match = q_match + normal_match
                assert len(match) == 1
                match = match[0].strip()
                assert not match.endswith(",")

                f, v = tuple(match.split("="))
                assert f == model_p
                assert f not in features
                features[f] = v

        # # namespace = [thing.strip() for thing in namespace.split(",")]
        # for thing in namespace:
        #     print(f"spliting {thing} on =")
        #     f, v = tuple(thing.split("="))
        #     assert f not in features
        #     features[f] = v
    return features

def read_rnn_params(f):
    params = {}
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    for line in lines:
        p, v = tuple(line.split("="))

        if v.endswith("\""):
            assert v.startswith("\"")
        if v.startswith("\""):
            assert v.endswith("\"")
            v = v[1:-1]
        
        params[p] = v

    return params




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="{COG_LANG_PAIR}_RNN-{RNN_HYPERPARAMS_ID}_S-{SEED}")
    parser.add_argument("--cog_lang_pair", default="fr-mfe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rnn_hyp_params_dir", default="/home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams")
    parser.add_argument("--slurm_outputs", default="/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    check_hyp_params(
        dir_template=args.dir,
        cog_lang_pair=args.cog_lang_pair,
        seed=args.seed,
        rnn_hyp_params_dir=args.rnn_hyp_params_dir,
        slurm_outputs_dir=args.slurm_outputs
    )