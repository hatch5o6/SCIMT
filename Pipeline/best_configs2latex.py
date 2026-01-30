import pandas as pd
import argparse
from decimal import Decimal, ROUND_HALF_UP

def main(best_configs):
    print("Entering main()")
    assert best_configs.endswith(".xlsx")
    print("reading best configs")
    df = pd.read_excel(best_configs)
    latex_path = best_configs[:-4] + "latex.txt"

    print('removing odd rows')
    # removing odd rows (because they're empty)
    odd_idxs = sorted([i for i in range(df.index.stop) if i % 2 != 0], reverse=True)
    for idx in odd_idxs:
        df = df.drop(idx)

    print("dropping unnecessary columns")
    assert list(df["enc_layer"]) != list(df["learning_rate"]) # sanity check for '==' operator
    assert list(df["enc_layer"]) == list(df["dec_layer"])
    assert list(df["enc_emb_dim"]) == list(df["dec_emb_dim"])
    assert list(df["enc_hid_dim"]) == list(df["dec_hid_dim"])
    df = df.drop(columns=["CRITERIA", "RNN_ID", "model_type", "attention", "dec_layer", "dec_emb_dim", "dec_hid_dim", "dropout", "learning_rate", "TRAIN / VAL / TEST SIZE"])
    print("renaming columns")
    df = df.rename(columns={"LANG": "Cognate Pair", "enc_layer": "Layers", "enc_emb_dim": "Emb. Dim.", "enc_hid_dim": "Hid. Dim.", "batch_size": "Batch Size", "BLEU": "charBLEU"})
    df["Cognate Pair"] = df["Cognate Pair"].apply(lambda x: f"\\textit{{{x}}}")

    print("converting to ints")
    for col in ["Layers", "Emb. Dim.", "Hid. Dim.", "Batch Size"]:
        df[col] = df[col].astype('int64')
    
    df = df.map(round_bleu)
    df = df.map(replace_dash)

    bold_columns = {col: f"\\textbf{{{col}}}" for col in df.columns}
    df = df.rename(columns=bold_columns)
    print("to latex")
    df.to_latex(buf=latex_path, index=False, escape=False, column_format='lrrrrr')

def replace_dash(x):
    return(
        x.replace("-", "/")
        if isinstance(x, str) else x
    )

def round_bleu(x):
    return (
        Decimal(str(x))
        .quantize(Decimal("0.00"), rounding=ROUND_HALF_UP)
        if isinstance(x, float) else x
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_configs", "-b", help=".xlsx file")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args.best_configs)
