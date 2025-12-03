import argparse
import os
import xlsxwriter
import json

def main(
    results_dir,
    include_lang_pairs,
    include_model_types
):
    for l in include_lang_pairs:
        l_dir = os.path.join(results_dir, l)
        write_sheet(l_dir, include_model_types)

def write_sheet(l_dir, include_model_types):
    workbook = xlsxwriter.Workbook(os.path.join(l_dir, "scores.xlsx"))
    header_format = workbook.add_format({'bold': True, 'bg_color': "#f2f2f2"})
    worksheet = workbook.add_worksheet()
    header = {"model": 0, "BLEU": 1, "chrF": 2, "test_data": 3, "val_data": 4, "checkpoint": 5}
    for h, c in header.items():
        worksheet.write(0, c, h, header_format)

    r = 1
    for model_dir in os.listdir(l_dir):
        model_dir_path = os.path.join(l_dir, model_dir)
        if not os.path.isdir(model_dir_path): continue
        prefix = model_dir.split(".")[0]
        if prefix not in include_model_types: continue

        scores_file = os.path.join(model_dir_path, "predictions/all_scores.json")
        scores_json = read_json(scores_file)
        best_scores = scores_json["BEST_VAL_BLEU_CHECKPOINT"]
        results = {"model": model_dir, "BLEU": best_scores["test_BLEU"], "chrF": best_scores["test_chrF"], "test_data": scores_json["TEST_DATA"], "val_data": scores_json["VAL_DATA"], "checkpoint": best_scores["checkpoint"]}

        for key, value in results.items():
            c_idx = header[key]
            worksheet.write(r, c_idx, value)
        
        r += 1
    worksheet.autofit()
    workbook.close()

def read_json(f):
    with open(f) as inf:
        data = json.load(inf)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--NMT_results_dir", "-d", default="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates")
    parser.add_argument("--include_lang_pairs", "-l", default="an-en,as-hi,bem-en,bho-as,bho-hi,djk-en,ewe-en,fon-fr,hsb-de,mfe-en,aeb-en,apc-en", help="comma-delimited list of NMT lang pairs")
    parser.add_argument("--include_model_types", "-m", default="FINETUNE,NMT,AUGMENT", help="Model types, comma-delimited list")
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
        include_model_types=include_model_types
    )

