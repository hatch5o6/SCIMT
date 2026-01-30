import pandas as pd
import argparse
from collections import Counter
import json
from tqdm import tqdm

def main(val_scores_xlsx, lang_pairs, out):
    print("reading dfs")
    data = {lp: pd.read_excel(val_scores_xlsx, sheet_name=lp) for lp in tqdm(lang_pairs)}
    results_runktrue = {}
    results_runkfalse = {}
    print("analyzing")
    for lang_pair, df in tqdm(data.items()):
        runktrue_difs = get_difs(df, prefix="RUNK=True")
        runkfalse_difs = get_difs(df, prefix="RUNK=False")

        assert lang_pair not in results_runktrue
        results_runktrue[lang_pair] = runktrue_difs

        assert lang_pair not in results_runkfalse
        results_runkfalse[lang_pair] = runkfalse_difs
    
    assert out.endswith(".json")
    out_runktrue = out[:-4] + ".runktrue.json"
    out_runkfalse = out[:-4] + ".runkfalse.json"
    with open(out_runktrue, "w") as outf:
        outf.write(json.dumps(results_runktrue, ensure_ascii=False, indent=2))
    with open(out_runkfalse, "w") as outf:
        outf.write(json.dumps(results_runkfalse, ensure_ascii=False, indent=2))

def get_difs(df, prefix="RUNK=True"):
    assert prefix in ["RUNK=True", "RUNK=False"]
    abs_difs_ct = Counter()
    matches = 0
    total = 0
    for idx, row in df.iterrows():
        eval_bleu = row[f"{prefix} eval BLEU"]
        fairseq_bleu = row[f"{prefix} fairseq BLEU"]
        match = row[f"{prefix} match?"]

        assert isinstance(eval_bleu, float), f"eval_bleu: {type(eval_bleu), eval_bleu}"
        assert isinstance(fairseq_bleu, float), f"eval_bleu: {type(fairseq_bleu), fairseq_bleu}"
        assert isinstance(match, bool)

        abs_dif = round(abs(eval_bleu - fairseq_bleu), 3)
        abs_difs_ct[abs_dif] += 1

        if match == True:
            matches += 1
        else:
            # assert match == "FALSE", f"match == '{match}' ({type(match)})"
            assert match == False
        total += 1
    
    assert total == 288

    abs_difs_dist = {
        k: str((v, percent(v, total)))
        for k, v in abs_difs_ct.items()
    }
    abs_difs_dist["matches"] = str((matches, percent(matches, total)))
    return abs_difs_dist
    
def percent(num1, num2):
    return round((num1 / num2) * 100, 2)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_scores", default="/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_val_scores_jan27.xlsx")
    parser.add_argument("--lang_pairs", default="bn-as,djk-en,en-djk,es-an,fon-ewe,ewe-fon,fr-mfe,lua-bem,bho-hi,ar-aeb,hi-bho,ar-apc")
    parser.add_argument("--out", default="/home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_val_scores_jan27.difs.json")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    lang_pairs = [item.strip() for item in args.lang_pairs.split(",")]
    main(args.val_scores, lang_pairs, args.out)