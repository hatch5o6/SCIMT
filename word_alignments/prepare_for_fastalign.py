import nltk
from nltk.tokenize import word_tokenize
import spacy
es_nlp = spacy.load('es_core_news_sm', exclude=["tagger", "parser", "ner", "lemmatizer", "textcat", "custom", "entity_linker", "entity_ruler", "textcat_multilabel", "trainable_lemmatizer", "morphologizer", "attribute_ruler", "senter", "sentencizer", "tok2vec", "transformer"])
nlp = {
    "es": es_nlp,
    "an": es_nlp,
    "oc": es_nlp,
    "ast": es_nlp
}
import argparse
from tqdm import tqdm

word_tokenize_langs = {
    "de": "german",
    "hsb": "czech",
    "ce": "czech",
    "en": "english",
    "fra": "french",
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
    
    if src_lang in nlp:
        src_tokenize = spacy_tokenize
    elif src_lang in word_tokenize_langs:
        src_tokenize = nltk_tokenize
    else:
        src_tokenize = nltk_tokenize
    
    if tgt_lang in nlp:
        tgt_tokenize = spacy_tokenize
    elif tgt_lang in word_tokenize_langs:
        tgt_tokenize = nltk_tokenize
    else:
        tgt_tokenize = nltk_tokenize

    for src, tgt in tqdm(pairs):
        # print("src")
        tokenized_src = src_tokenize(src, lang=src_lang)
        # print("tgt")
        tokenized_tgt = tgt_tokenize(tgt, lang=tgt_lang)
        formatted = tokenized_src.strip() + " ||| " + tokenized_tgt.strip()
        formatted_pairs.append(formatted)
    
    print("Writing prepare_for_fastalign.py output to", out_f)
    with open(out_f, "w") as outf:
        outf.write("\n".join(formatted_pairs) + "\n")

def spacy_tokenize(line, lang):
    # print(f"spacy_tokenize lang={lang}")
    doc = nlp[lang](line)
    tokens = [tok.text for tok in doc]
    return " ".join(tokens)

def nltk_tokenize(line, lang):
    global word_tokenize_langs
    wtlang = word_tokenize_langs.get(lang, "english")
    # print(f"nltk_tokenize lang={wtlang}")
    tokens = word_tokenize(line.strip(), language=wtlang)
    return " ".join(tokens)

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
    print("------------------\n\n")
    return args

if __name__ == "__main__":
    print("prepare_for_fastlign.py")
    args = get_args()
    prep(args.src, args.tgt, args.out)