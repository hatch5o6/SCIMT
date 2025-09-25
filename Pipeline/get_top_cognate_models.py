import argparse
import pandas as pd
import os


def main(
    hyperparam_search_results_dir,
    pair
):
    f = os.path.join(hyperparam_search_results_dir, f"{pair}.results.xlsx")
    df = pd.read_excel(f)

    models = []
    for idx, row in df.iterrows():
        rnn_id = row["RNN_ID"]
        model_type = row["model_type"]
        bleu = float(row["BLEU"])
        chrf = float(row["chrF"])
        models.append((bleu, chrf, rnn_id, model_type))
    
    models.sort(reverse=True)

    print("TOP MODELS:")
    for bleu, chrf, rnn_id, model_type in models[:10]:
        if model_type == "bigru":
            prefix = "RNN"
        else:
            assert model_type == "SMT"
            prefix = "SMT"

        print(f"\t{prefix}-{rnn_id}: BLEU ({bleu}), chrF ({chrf})")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyperparam_search_results", default="/home/hatch5o6/Cognate/code/Pipeline/hyperparam_search_results/CUR_09_19_2025_10:24")
    parser.add_argument("--pair", help="e.g. 'fr-mfe'")
    args = parser.parse_args()
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t--{k}=`{v}`")
    print("\n\n")
    return args

if __name__ == "__main__":
    print("#############################")
    print("# get_top_cognate_models.py #")
    print("#############################")
    args = get_args()
    main(
        hyperparam_search_results_dir=args.hyperparam_search_results,
        pair=args.pair
    )
    