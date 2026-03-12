import argparse
import os
import xlsxwriter
import json
import csv
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
import math

ONE_SET = {
    "bho-hi": ("hi", "bho", "hi"),
    "djk-en": ("en", "djk", "en")
}
THREE_SET = {
    "aeb-en": ("ar", "aeb", "en"),
    "apc-en": ("ar", "apc", "en"),
    "an-en": ("es", "an", "en"),
    "anx-enx": ("esx", "anx", "enx"),
    "as-hi": ("bn", "as", "hi"),
    "bem-en": ("lua", "bem", "en"),
    "bho-as": ("hi", "bho", "as"),
    "ewe-en": ("fon", "ewe", "en"),
    "fon-fr": ("ewe", "fon", "fr"),
    "mfe-en": ("fr", "mfe", "en"),
    "mfx-enx": ("frx", "mfx", "enx"),
    "mfy-eny": ("fry", "mfy", "eny"),
    "oc-en": ("fr", "oc", "en")
}

def main(
    results_dir,
    include_lang_pairs,
    include_model_types,
    nmt_data_params_log,
    sc_best_configs,
    latex_out,
    REVERSE=False
):
    all_l_results = {}
    for l in include_lang_pairs:
        l_dir = os.path.join(results_dir, l)
        if not os.path.exists(l_dir): continue

        l_results = write_sheet(l_dir, include_model_types, REVERSE=REVERSE)
        l_results = find_best_scores(l_results)
        assert l not in all_l_results
        all_l_results[l] = l_results
    sc_best_cf = read_sc_best_config(sc_best_configs)
    nmt_log = read_nmt_log(nmt_data_params_log)
    make_latex(all_l_results, latex_out, sc_best_cf, nmt_log)

def read_nmt_log(f):
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = {col: c for c, col in enumerate(rows[0])}
    data = rows[1:]
    data_dict = {}
    for row in data:
        if not is_empty_row(row):
            config_file = row[header["config"]]
            num_train = int(row[header["train"]].replace(",", ""))
            num_val = int(row[header["val"]].replace(",", ""))
            num_test = int(row[header["test"]].replace(",", ""))
            assert config_file not in data_dict
            data_dict[config_file] = [num_train, num_val, num_test]
    return data_dict

def is_empty_row(row):
    for elem in row:
        if elem.strip() != "":
            return False
    return True

def read_sc_best_config(f):
    df = pd.read_excel(f)
    data_dict = {}
    for idx, row in df.iterrows():
        lang = row["LANG"]
        if isinstance(lang, float):
            assert math.isnan(lang)
            continue
        else:
            assert isinstance(lang, str)
            lang = lang.strip()
        train_val_test = row["TRAIN / VAL / TEST SIZE"]
        train, val, test = [int(item.replace(",", "").strip()) for item in train_val_test.split("/")]
        assert lang not in data_dict
        data_dict[lang] = [train, val, test]
    return data_dict

def find_best_scores(l_results):
    best_BLEU, best_chrF = None, None
    best_BLEU_model, best_chrF_model = None, None
    for model_dir, results in l_results.items():
        l_results[model_dir]["best_BLEU"] = False
        l_results[model_dir]["best_chrF"] = False
        if best_BLEU is None:
            assert best_BLEU_model is None
            best_BLEU = results["BLEU"]
            best_BLEU_model = model_dir
        elif results["BLEU"] > best_BLEU:
            best_BLEU = results["BLEU"]
            best_BLEU_model = model_dir

        if best_chrF is None:
            assert best_chrF_model is None
            best_chrF = results["chrF"]
            best_chrF_model = model_dir
        elif results["chrF"] > best_chrF:
            best_chrF = results["chrF"]
            best_chrF_model = model_dir
    l_results[best_BLEU_model]["best_BLEU"] = True
    l_results[best_chrF_model]["best_chrF"] = True
    return l_results
    
def make_latex(all_l_results, latex_out, sc_best_cf, nmt_log):
    assert latex_out.endswith(".txt")
    df_one_set = make_df(all_l_results, sc_best_cf, nmt_log, scen_set=ONE_SET)
    df_three_set = make_df(all_l_results, sc_best_cf, nmt_log, scen_set=THREE_SET)

    write_latex(df_one_set, latex_out[:-3] + "one_set.txt")
    write_latex(df_three_set, latex_out[:-3] + "three_set.txt")

def write_latex(df, latex_out):
    if os.path.exists(latex_out):
        os.remove(latex_out)
    column_format = "l" + "".join(["r" for i in range(len(df.columns))])
    df.to_latex(buf=latex_out, index=False, escape=False, column_format=column_format)

def get_scen_string(scen):
    pl, cl, tl = scen
    return f"${pl}/{cl} \\rightarrow {tl}$", pl, cl, tl

def make_df(all_l_results, sc_best_cf, nmt_log, scen_set):
    SCEN = "\\textbf{Scenario}"
    TRAIN_PL_TL = "\\textbf{\\textit{n}P→T}"
    TRAIN_CL_TL = "\\textbf{\\textit{n}C→T}"
    TRAIN_PL_CL = "\\textbf{\\textit{n}P/C}"
    CL_TL = "\\textbf{C→T}"
    CLp_TL = "\\textbf{C'→T}"
    PRE_PL_TL = "\\textbf{P→T}"
    PRE_PLp_TL = "\\textbf{P'→T}"
    FIN_PL_TL = "\\textbf{P→T>>C→T}"
    FIN_PLp_TL = "\\textbf{P'→T>>C→T}"

    if scen_set == ONE_SET:
        TRAIN_PL_TL = "\\textbf{\\textit{n}T'→T}"
        TRAIN_PL_CL = "\\textbf{\\textit{n}T/C}"
        FIN_PLp_TL = "\\textbf{T'→T>>C→T}"

    data = {
        SCEN:[],
        TRAIN_PL_CL: [],
        TRAIN_PL_TL: [],
        TRAIN_CL_TL: [],
        CL_TL: [],
        CLp_TL: [],
        PRE_PL_TL: [], 
        PRE_PLp_TL: [],
        FIN_PL_TL: [],
        FIN_PLp_TL: []
    }
    for l, l_results in all_l_results.items():
        if l not in scen_set: continue
        scen, pl, cl, tl = get_scen_string(scen_set[l])
        data[SCEN].append(scen)

        pl_cl_pairs = sc_best_cf[f"{pl}-{cl}"][0]
        if scen_set == THREE_SET:
            pl_tl_pairs = nmt_log[f"{cl}-{tl}/PRETRAIN.{pl}-{tl}.yaml"][0]
            cl_tl_pairs = nmt_log[f"{cl}-{tl}/FINETUNE.{pl}-{tl}>>{cl}-{tl}.yaml"][0]
        else:
            assert scen_set == ONE_SET
            pl_tl_pairs = nmt_log[f"{cl}-{tl}/PRETRAIN.SC_{pl}2{cl}-{tl}.ATT.yaml"][0]
            cl_tl_pairs = nmt_log[f"{cl}-{tl}/FINETUNE.SC_{pl}2{cl}-{tl}>>{cl}-{tl}.ATT.yaml"][0]

        data[TRAIN_PL_CL].append(round_letter(pl_cl_pairs))
        data[TRAIN_PL_TL].append(round_letter(pl_tl_pairs))
        data[TRAIN_CL_TL].append(round_letter(cl_tl_pairs))

        for model_dir, results in l_results.items():
            BLEU = str(float(round_bleu(results["BLEU"])))
            chrF = str(float(round_bleu(results["chrF"])))

            if results["best_BLEU"]:
                BLEU = f"\\textbf{{{BLEU}}}"
            if results["best_chrF"]:
                chrF = f"\\textbf{{{chrF}}}"

            score_str = f"{BLEU}/{chrF}"
            if model_dir.startswith("NMT.SC"):
                data[CLp_TL].append(score_str)
            elif model_dir.startswith("NMT"):
                data[CL_TL].append(score_str)
            elif model_dir.startswith("PRETRAIN.SC"):
                data[PRE_PLp_TL].append(score_str)
            elif model_dir.startswith("PRETRAIN"):
                data[PRE_PL_TL].append(score_str)
            elif model_dir.startswith("FINETUNE.SC"):
                data[FIN_PLp_TL].append(score_str)
            elif model_dir.startswith("FINETUNE"):
                data[FIN_PL_TL].append(score_str)
        
        data_len = len(data[SCEN])
        for header, values in data.items():
            assert len(values) in [0, data_len]
    headers_to_pop = [header for header, values in data.items() if len(values) == 0]
    for header in headers_to_pop:
        data.pop(header)
    df = pd.DataFrame.from_dict(data)
    return df
    
def round_letter(x):
    if x >= 1000000:
        return round_M(x)
    else:
        return round_K(x)

def round_K(x):
    return f"{(x + 500) // 1000}K"

def round_M(x):
    return f"{(x + 500000) // 1000000}M"

def round_bleu(x):
    return (
        Decimal(str(x))
        .quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        if isinstance(x, float) else x
    )

def write_sheet(l_dir, include_model_types, REVERSE=False):
    workbook = xlsxwriter.Workbook(os.path.join(l_dir, "scores.xlsx"))
    header_format = workbook.add_format({'bold': True, 'bg_color': "#f2f2f2"})
    worksheet = workbook.add_worksheet()
    header = {"model": 0, "BLEU": 1, "chrF": 2, "test_data": 3, "val_data": 4, "checkpoint": 5}
    for h, c in header.items():
        worksheet.write(0, c, h, header_format)

    all_results = {}
    r = 1
    for model_dir in os.listdir(l_dir):
        if REVERSE == False and "REVERSE_TRIAL" in model_dir:
            continue
        elif REVERSE == True and "REVERSE_TRIAL" not in model_dir:
            continue

        model_dir_path = os.path.join(l_dir, model_dir)
        if not os.path.isdir(model_dir_path): continue
        prefix = model_dir.split(".")[0]
        if prefix not in include_model_types: continue

        scores_file = os.path.join(model_dir_path, "predictions/all_scores.json")
        scores_json = read_json(scores_file)
        best_scores = scores_json["BEST_VAL_BLEU_CHECKPOINT"]
        results = {
            "model": model_dir, 
            "BLEU": best_scores["test_BLEU"], 
            "chrF": best_scores["test_chrF"], 
            "test_data": scores_json["TEST_DATA"], 
            "val_data": scores_json["VAL_DATA"], 
            "checkpoint": best_scores["checkpoint"]
        }

        for key, value in results.items():
            c_idx = header[key]
            worksheet.write(r, c_idx, value)
        
        assert model_dir not in all_results
        all_results[model_dir] = results
        r += 1
    worksheet.autofit()
    workbook.close()
    return all_results

def read_json(f):
    with open(f) as inf:
        data = json.load(inf)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NMT_results_dir", "-d", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates")
    parser.add_argument("--include_lang_pairs", "-l", default="an-en,as-hi,bem-en,bho-as,bho-hi,djk-en,ewe-en,fon-fr,hsb-de,mfe-en,aeb-en,apc-en", help="comma-delimited list of NMT lang pairs")
    parser.add_argument("--include_model_types", "-m", default="FINETUNE,NMT", help="Model types, comma-delimited list")
    parser.add_argument("--sc_best_configs", "-c", default="/home/hatch5o6/Cognate/code/Pipeline/hyperparam_search_results/CUR_01_20_2026_15:23/best_configs.xlsx", help="best_configs.xlsx file from the sc hyperparameter search")
    parser.add_argument("--nmt_data_params_log", "-n", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/data_params_log.csv", help="data_params_log.csv file from NMT experiments")
    parser.add_argument("--REVERSE", action="store_true", help="if passed, will do table of scores for REVERSE NMT directions")
    parser.add_argument("--latex_out", "-o")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t-{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    args = get_args()
    include_model_types = [m.strip() for m in args.include_model_types.split(",")]
    for m in include_model_types:
        assert m in ["NMT", "FINETUNE", "PRETRAIN", "AUGMENT"], f"`{m}` is not a valid model type!"

    include_lang_pairs = [l.strip() for l in args.include_lang_pairs.split(",")]
    # for l in include_lang_pairs:
    #     assert l in ["an-en", "as-hi", "bem-en", "bho-as", "bho-hi", "djk-en", "ewe-en", "fon-fr", "hsb-de", "mfe-en", "aeb-en", "apc-en"], f"`{l}` is not a valid lang pair!"
    
    main(
        results_dir=args.NMT_results_dir,
        include_lang_pairs=include_lang_pairs,
        include_model_types=include_model_types,
        nmt_data_params_log=args.nmt_data_params_log,
        sc_best_configs=args.sc_best_configs,
        latex_out=args.latex_out,
        REVERSE=args.REVERSE
    )

