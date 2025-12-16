import argparse
import os
import xlsxwriter
from xlsxwriter.color import Color
from tqdm import tqdm
import datetime

def compile(
    langs,
    rnn_hyperparams_dir,
    COPPERMT,
    seed,
    out_dir,
    tag
):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    now = datetime.datetime.now()
    timestamp = now.strftime('%m_%d_%Y_%H:%M')
    tag_dir = os.path.join(out_dir, f"{tag}_{timestamp}")
    if not os.path.exists(tag_dir):
        os.mkdir(tag_dir)

    BEST_LANG_CONFIGS = {}

    header = {
        "RNN_ID": 0,
        "model_type": 1,
        "attention": 2, 
        "enc_layer": 3,
        "enc_emb_dim": 4, 
        "enc_hid_dim": 5,
        "dec_layer": 6,
        "dec_emb_dim": 7,
        "dec_hid_dim": 8,
        "batch_size": 9,
        "dropout": 10,
        "learning_rate": 11,
        "BLEU": 12,
        "chrF": 13
    }

    for lang in langs:
        print("LANG:", lang)
        out_f = os.path.join(tag_dir, f"{lang}.results.xlsx")
        if os.path.exists(out_f):
            print("Removing:", out_f)
            os.remove(out_f)
        
        lang_workbook = xlsxwriter.Workbook(out_f)
        header_format = lang_workbook.add_format({"bold": True})
        rnn_id_format = lang_workbook.add_format({"bold": True, "bg_color": Color("#DFDFE1"), "align": "right"})
        chrF_format = lang_workbook.add_format({"bg_color": Color("#E2DDAC")})
        best_chrF_format = lang_workbook.add_format({"bold": True, "bg_color": Color("#E3D970")})
        BLEU_format = lang_workbook.add_format({"bg_color": Color("#AFC7F7")})
        best_BLEU_format = lang_workbook.add_format({"bold": True, "bg_color": Color("#78A3FA")})
        param_format = lang_workbook.add_format({"align": "right"})
        
        assert lang not in BEST_LANG_CONFIGS
        BEST_LANG_CONFIGS[lang] = {"BLEU": {}, "chrF": {}}

        worksheet = lang_workbook.add_worksheet()
        for key, idx in header.items():
            worksheet.write(0, idx, key, header_format)

        visited_rnn_ids = set()
        BEST_BLEU = None
        BEST_chrF = None
        for f in tqdm(os.listdir(rnn_hyperparams_dir) + ["SMT"]):
            if f == "SMT":
                results_rnn_params = {
                    "model_type": "SMT",
                    "attention": "n/a",
                    "enc_layer": "n/a",
                    "enc_emb_dim": "n/a",
                    "enc_hid_dim": "n/a",
                    "dec_layer": "n/a",
                    "dec_emb_dim": "n/a",
                    "dec_hid_dim": "n/a",
                    "batch_size": "n/a",
                    "dropout": "n/a",
                    "learning_rate": "n/a",
                    "share_encoder": "n/a",
                    "share_decoder": "n/a"
                }
                rnn_id = get_smt_id(rnn_hyperparams_dir=rnn_hyperparams_dir)
                assert rnn_id not in visited_rnn_ids
                assert int(rnn_id) not in visited_rnn_ids
                COPPERMT_results_dir = os.path.join(COPPERMT, f"{src_lang}_{tgt_lang}_SMT-null_S-{seed}")
                # scores_f = os.path.join(COPPERMT_results_dir, f"workspace/reference_models/statistical/{seed}/{src_lang}_{tgt_lang}/out/??")
                scores_f = os.path.join(COPPERMT_results_dir, f"inputs/split_data/{src_lang}_{tgt_lang}/{seed}/test_{src_lang}_{tgt_lang}.{tgt_lang}.hyp.scores.txt")
            else:
                f_path = os.path.join(rnn_hyperparams_dir, f)
                if os.path.isdir(f_path): continue
                if f == "manifest.json": continue

                print("checking that", f, "ends with .rnn.txt")
                assert f.endswith(".rnn.txt")
                split_f = f.split(".")
                assert len(split_f) == 3
                rnn_id = split_f[0]
                assert isinstance(rnn_id, str)
                assert rnn_id not in visited_rnn_ids
                assert int(rnn_id) not in visited_rnn_ids

                rnn_params = read_rnn_params_f(f_path)

                src_lang, tgt_lang = tuple(lang.split("-"))
                COPPERMT_results_dir = os.path.join(COPPERMT, f"{src_lang}_{tgt_lang}_RNN-{rnn_id}_S-{seed}")
                results_rnn_params_f = os.path.join(COPPERMT_results_dir, f"inputs/parameters/bilingual_default/default_parameters_rnn_{lang}.txt")
                results_rnn_params = read_rnn_params_f(results_rnn_params_f)

                print("asserting rnn_params == results_rnn_params")
                assert rnn_params == results_rnn_params
                print("\tpassed :)")
                scores_f = os.path.join(COPPERMT_results_dir, f"workspace/reference_models/bilingual/rnn_{src_lang}-{tgt_lang}/0/results/test_selected_checkpoint_{src_lang}_{tgt_lang}.{tgt_lang}/generate-test.hyp.scores.txt")
            
            
            BLEU, chrF = read_scores(scores_f)

            assert rnn_id not in visited_rnn_ids
            assert int(rnn_id) not in visited_rnn_ids
            visited_rnn_ids.add(rnn_id)

            worksheet.write(int(rnn_id) + 1, header["RNN_ID"], rnn_id, rnn_id_format)
            for param, param_val in results_rnn_params.items():
                if param in ["share_encoder", "share_decoder"]: continue
                worksheet.write(int(rnn_id) + 1, header[param], param_val, param_format)
            worksheet.write(int(rnn_id) + 1, header["BLEU"], BLEU, BLEU_format)
            worksheet.write(int(rnn_id) + 1, header["chrF"], chrF, chrF_format)

            if BEST_BLEU is None:
                BEST_BLEU = (int(rnn_id) + 1, header["BLEU"], BLEU, BLEU_format)
            else:
                if BLEU > BEST_BLEU[2]:
                    BEST_BLEU = (int(rnn_id) + 1, header["BLEU"], BLEU, BLEU_format)
                    BEST_LANG_CONFIGS[lang]["BLEU"]["params"] = results_rnn_params
                    BEST_LANG_CONFIGS[lang]["BLEU"]["BLEU"] = BLEU
                    BEST_LANG_CONFIGS[lang]["BLEU"]["chrF"] = chrF
                    BEST_LANG_CONFIGS[lang]["BLEU"]["rnn_id"] = rnn_id
            
            if BEST_chrF is None:
                BEST_chrF = (int(rnn_id) + 1, header["chrF"], chrF, chrF_format)
            else:
                if chrF > BEST_chrF[2]:
                    BEST_chrF = (int(rnn_id) + 1, header["chrF"], chrF, chrF_format)
                    BEST_LANG_CONFIGS[lang]["chrF"]["params"] = results_rnn_params
                    BEST_LANG_CONFIGS[lang]["chrF"]["chrF"] = chrF
                    BEST_LANG_CONFIGS[lang]["chrF"]["BLEU"] = BLEU
                    BEST_LANG_CONFIGS[lang]["chrF"]["rnn_id"] = rnn_id

        worksheet.write(BEST_BLEU[0], BEST_BLEU[1], BEST_BLEU[2], best_BLEU_format)
        worksheet.write(BEST_chrF[0], BEST_chrF[1], BEST_chrF[2], best_chrF_format)

        worksheet.autofit()
        lang_workbook.close()

    best_out_f = os.path.join(tag_dir, "best_configs.xlsx")
    best_workbook = xlsxwriter.Workbook(best_out_f)
    best_header_format = best_workbook.add_format({"bold": True})
    best_rnn_id_format = best_workbook.add_format({"bold": True, "bg_color": Color("#DFDFE1"), "align": "right"})
    best_best_chrF_format = best_workbook.add_format({"bold": True, "bg_color": Color("#E3D970")})
    best_best_BLEU_format = best_workbook.add_format({"bold": True, "bg_color": Color("#78A3FA")})
    best_param_format = lang_workbook.add_format({"align": "right"})
    
    best_worksheet = best_workbook.add_worksheet()
    best_worksheet.write(0, 0, "LANG", best_header_format)
    best_worksheet.write(0, 1, "CRITERIA", best_header_format)
    for key, idx in header.items():
        best_worksheet.write(0, idx + 2, key, best_header_format)
    for lx, (lang, criteria_configs) in enumerate(BEST_LANG_CONFIGS.items()):
        for cx, (criteria, configs) in enumerate(criteria_configs.items()):
            best_worksheet.write((lx * 2) + cx + 1, 0, lang, best_header_format)
            best_worksheet.write((lx * 2) + cx + 1, 1, criteria, best_header_format)
            for key, idx in header.items():
                if key in configs["params"]:
                    best_worksheet.write((lx * 2) + cx + 1, idx + 2, configs["params"][key], best_param_format)
                elif key == "RNN_ID":
                    best_worksheet.write((lx * 2) + cx + 1, idx + 2, configs["rnn_id"], best_rnn_id_format)
                elif key == "chrF":
                    best_worksheet.write((lx * 2) + cx + 1, idx + 2, configs["chrF"], best_best_chrF_format)
                elif key == "BLEU":
                    best_worksheet.write((lx * 2) + cx + 1, idx + 2, configs["BLEU"], best_best_BLEU_format)
    best_worksheet.autofit()
    best_workbook.close()
    

def get_smt_id(rnn_hyperparams_dir):
    max_rnn_id = None
    for f in os.listdir(rnn_hyperparams_dir):
        f_path = os.path.join(rnn_hyperparams_dir, f)
        if os.path.isdir(f_path): continue
        if f == "manifest.json": continue

        split_f = f.split(".")
        assert len(split_f) == 3
        rnn_id = int(split_f[0])
        if max_rnn_id is None:
            max_rnn_id = rnn_id
        else:
            if rnn_id > max_rnn_id:
                max_rnn_id = rnn_id

    smt_rnn_id = max_rnn_id + 1
    return smt_rnn_id

def read_scores(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    BLEU = None
    chrF = None
    for l, line in enumerate(lines):
        if l == 0:
            assert line == "Scores:"
        elif l == 1:
            assert line.startswith("\tREF: ")
        elif l == 2:
            assert line.startswith("\tHYP: ")
        elif l == 3:
            assert line == ""
        elif l == 4:
            assert line.startswith("BLEU: ")
            assert BLEU == None
            BLEU = float(line.split("BLEU: ")[1])
        elif l == 5:
            assert line.startswith("chrF: ")
            assert chrF == None
            chrF = float(line.split("chrF: ")[1])
    assert BLEU is not None
    assert chrF is not None
    return BLEU, chrF


def read_rnn_params_f(f):
    print("READING PARAMS f", f)
    params = {}
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    for line in lines:
        f, v = tuple(line.split("="))
        if v.endswith("\""):
            assert v.startswith("\"")
        if v.startswith("\""):
            assert v.endswith("\"")
            v = v[1:-1]
        assert f not in params
        params[f] = v
    return params

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", help="comma-delimited list")
    parser.add_argument("--rnn_hyperparams_dir", default="/home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams")
    parser.add_argument("--COPPERMT", default="/home/hatch5o6/nobackup/archive/CopperMT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="/home/hatch5o6/Cognate/code/Pipeline/hyperparam_search_results")
    parser.add_argument("--tag")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    langs = [l.strip() for l in args.langs.split(",")]
    compile(
        langs=langs,
        rnn_hyperparams_dir=args.rnn_hyperparams_dir,
        COPPERMT=args.COPPERMT,
        seed=args.seed,
        out_dir=args.out,
        tag=args.tag
    )