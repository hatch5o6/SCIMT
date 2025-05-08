import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import argparse
from tqdm import tqdm

word_tokenize_langs = {
    "de": "german",
    "hsb": "czech"
}

def prep(
    src_f,
    tgt_f,
    out_f
):
    src = read_file(src_f)
    tgt = read_file(tgt_f)
    src_lang = src_f.split(".")[-1]
    tgt_lang = tgt_f.split(".")[-1]
    assert len(src) == len(tgt)

    pairs = list(zip(src, tgt))
    formatted_pairs = []
    for src, tgt in tqdm(pairs):
        # print("src")
        tokenized_src = tokenize(src, lang=src_lang)
        # print("tgt")
        tokenized_tgt = tokenize(tgt, lang=tgt_lang)
        formatted = tokenized_src.strip() + " ||| " + tokenized_tgt.strip()
        formatted_pairs.append(formatted)
    
    with open(out_f, "w") as outf:
        outf.write("\n".join(formatted_pairs) + "\n")

def tokenize(line, lang):
    global word_tokenize_langs
    wtlang = word_tokenize_langs[lang]
    # print(f"word_tokenize lang='{wtlang}'")
    return " ".join(word_tokenize(line, language=wtlang))

def read_file(f):
    with open(f) as inf:
        lines = [line.strip() for line in inf]
    return lines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--out", "-o", required=True)
    args = parser.parse_args()
    print("arguments")
    for k, v in vars(args).items():
        print(f"{k}: '{v}'")
    print("\n\n")
    return args

if __name__ == "__main__":
    args = get_args()
    prep(args.src, args.tgt, args.out)