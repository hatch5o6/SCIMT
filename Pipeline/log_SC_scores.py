import os
import csv
import argparse

def main(
    configs_dir,
    CopperMT_dir,
    out_f
):
    header = {
        "SC_MODEL_ID": 0, 
        "SC_MODEL_TYPE": 1,
        "RNN_HYPERPARAMS_ID": 2,
        "SEED": 3,
        "COGNATE_THRESH": 4,
        "LOG_P_THRESH": 5,
        "EDIT_DIST_TYPE": 6,
        "BLEU": 7,
        "chrF": 8,
    }
    next_h = max(header.values()) + 1

    with open(out_f, "w", newline='') as outf:
        writer = csv.writer(outf)
        prev_group_key = None
        for idxf, f in enumerate(os.listdir(configs_dir)):
            f_path = os.path.join(configs_dir, f)
            f_config = read_config(f_path)

            if idxf == 0:
                for k in f_config.keys():
                    if k not in header:
                        header[k] = next_h
                        next_h += 1
                write_header = [None for i in range(len(header))]
                for head, idx in header.items():
                    write_header[idx] = head
                writer.writerow(write_header)

            sc_model_id = f_config['SC_MODEL_ID']
            model_type = f_config["SC_MODEL_TYPE"]
            assert model_type in ["SMT", "RNN"], f"Invalid model type in `{f_path}`: `{model_type}`"

            rnn_hyp_id = f_config["RNN_HYPERPARAMS_ID"]

            group_key = "-".join(sc_model_id.split("-")[:3]) + "-" + str(rnn_hyp_id)
            if prev_group_key is not None and group_key != prev_group_key:
                writer.writerow([])

            seed = f_config["SEED"]
            model_coppermt_dir_name = f"{sc_model_id}_{model_type}-{rnn_hyp_id}_S-{seed}"
            model_coppermt_dir = os.path.join(CopperMT_dir, model_coppermt_dir_name)

            if model_type == "SMT":
                get_score = get_smt_score
            elif model_type == "RNN":
                get_score = get_rnn_score

            bleu, chrf = get_score(model_coppermt_dir, src_lang=f_config["SRC"], tgt_lang=f_config["TGT"], seed=seed)
            row = [None for i in range(len(header))]
            row[header["BLEU"]] = bleu
            row[header["chrF"]] = chrf

            for k, v in f_config.items():
                idxr = header[k]
                row[idxr] = v
            writer.writerow(row)

            prev_group_key = group_key


def get_smt_score(model_coppermt_dir, src_lang, tgt_lang, seed):
    scores_f = os.path.join(
        model_coppermt_dir,
        f"inputs/split_data/{src_lang}_{tgt_lang}/{seed}/test_{src_lang}_{tgt_lang}.{tgt_lang}.hyp.scores.txt"
    )
    return parse_score(scores_f)

def get_rnn_score(model_coppermt_dir, src_lang, tgt_lang, seed):
    scores_f = os.path.join(
        model_coppermt_dir,
        f"workspace/reference_models/bilingual/rnn_{src_lang}-{tgt_lang}/{seed}/results/test_selected_checkpoint_{src_lang}_{tgt_lang}.{tgt_lang}/generate-test.hyp.scores.txt"
    )
    return parse_score(scores_f)

def parse_score(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    assert len(lines) == 6
    bleu = None
    chrf = None
    for l, line in enumerate(lines):
        if l == 0:
            assert line == "Scores:"
        elif l == 1:
            assert line.startswith("\tREF: ")
        elif l == 2:
            assert line.startswith(f"\tHYP: ")
        elif l == 3:
            assert line == ""
        elif l == 4:
            assert line.startswith("BLEU: ")
            bleu = float(line.split("BLEU: ")[-1])
        elif l == 5:
            assert line.startswith("chrF: ")
            chrf = float(line.split("chrF: ")[-1])
    assert bleu is not None
    assert chrf is not None

    return bleu, chrf


def read_config(f):
    config = {}
    with open(f) as inf:
        for line in inf.readlines():
            line = line.strip()
            if line == "": continue
            if line.startswith("#"): continue

            assert line != ""
            assert "=" in line
            k = line.split("=")[0]
            v = "=".join(line.split("=")[1:])

            assert k not in config
            config[k] = v
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", required=True)
    parser.add_argument("--CopperMT_dir", default="/home/hatch5o6/nobackup/archive/CopperMT")
    parser.add_argument("--out")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.configs_dir, args.CopperMT_dir, args.out)