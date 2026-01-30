import argparse
import os
import xlsxwriter
import json
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP

ONE_SET = {
    "bho-hi": ("hi", "bho", "hi"),
    "djk-en": ("en", "djk", "en")
}
THREE_SET = {
    "aeb-en": ("ar", "aeb", "en"),
    "apc-en": ("ar", "apc", "en"),
    "an-en": ("es", "an", "en"),
    "as-hi": ("bn", "as", "hi"),
    "bem-en": ("lua", "bem", "en"),
    "bho-as": ("hi", "bho", "as"),
    "ewe-en": ("fon", "ewe", "en"),
    "fon-fr": ("ewe", "fon", "fr"),
    "mfe-en": ("fr", "mfe", "en")
}

def main(
    results_dir,
    include_lang_pairs,
    include_model_types,
    latex_out
):
    all_l_results = {}
    for l in include_lang_pairs:
        l_dir = os.path.join(results_dir, l)
        if not os.path.exists(l_dir): continue

        l_results = write_sheet(l_dir, include_model_types)
        l_results = find_best_scores(l_results)
        assert l not in all_l_results
        all_l_results[l] = l_results
    make_latex(all_l_results, latex_out)

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
    
def make_latex(all_l_results, latex_out):
    assert latex_out.endswith(".txt")
    df_one_set = make_df(all_l_results, scen_set=ONE_SET)
    df_three_set = make_df(all_l_results, scen_set=THREE_SET)

    write_latex(df_one_set, latex_out[:-3] + "one_set.txt")
    write_latex(df_three_set, latex_out[:-3] + "three_set.txt")

def write_latex(df, latex_out):
    if os.path.exists(latex_out):
        os.remove(latex_out)
    column_format = "l" + "".join(["r" for i in range(len(df.columns))])
    df.to_latex(buf=latex_out, index=False, escape=False, column_format=column_format)

def get_scen_string(scen):
    pl, cl, tl = scen
    return f"${pl}/{cl} \\rightarrow {tl}$"

def make_df(all_l_results, scen_set):
    SCEN = "\\textbf{PL/CL→TL}"
    CL_TL = "\\textbf{CL→TL}"
    CLp_TL = "\\textbf{CL'→TL}"
    PRE_PL_TL = "\\textbf{PL→TL}"
    PRE_PLp_TL = "\\textbf{PL'→TL}"
    FIN_PL_TL = "\\textbf{PL→TL >> CL→TL}"
    FIN_PLp_TL = "\\textbf{PL'→TL >> CL→TL}"
    data = {
        SCEN:[], 
        CL_TL: [],
        CLp_TL: [],
        PRE_PL_TL: [], 
        PRE_PLp_TL: [],
        FIN_PL_TL: [],
        FIN_PLp_TL: []
    }
    for l, l_results in all_l_results.items():
        if l not in scen_set: continue
        scen = get_scen_string(scen_set[l])
        data[SCEN].append(scen)
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
    

def round_bleu(x):
    return (
        Decimal(str(x))
        .quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        if isinstance(x, float) else x
    )

def write_sheet(l_dir, include_model_types):
    workbook = xlsxwriter.Workbook(os.path.join(l_dir, "scores.xlsx"))
    header_format = workbook.add_format({'bold': True, 'bg_color': "#f2f2f2"})
    worksheet = workbook.add_worksheet()
    header = {"model": 0, "BLEU": 1, "chrF": 2, "test_data": 3, "val_data": 4, "checkpoint": 5}
    for h, c in header.items():
        worksheet.write(0, c, h, header_format)

    all_results = {}
    r = 1
    for model_dir in os.listdir(l_dir):
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
    parser.add_argument("--include_model_types", "-m", default="FINETUNE,NMT,AUGMENT", help="Model types, comma-delimited list")
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
        latex_out=args.latex_out
    )

