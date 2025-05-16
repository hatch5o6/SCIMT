import argparse
import os
import shutil
from tqdm import tqdm
from string import punctuation
punctuation += "-—¡¿؟؛،٪»«›‹”“〞❮❯❛❟%."

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
nltk_tokenize_langs = {
    "de": "german",
    "hsb": "czech",
    "cs": "czech",
    "en": "english"
}

import spacy
es_nlp = spacy.load('es_core_news_sm', exclude=["tagger", "parser", "ner", "lemmatizer", "textcat", "custom", "entity_linker", "entity_ruler", "textcat_multilabel", "trainable_lemmatizer", "morphologizer", "attribute_ruler", "senter", "sentencizer", "tok2vec", "transformer"])
spacy_nlp = {
    "es": es_nlp,
    "an": es_nlp,
    "oc": es_nlp,
    "ast": es_nlp
}

EOS = "<EOS>"

# function for preparing HR lang data for preprocessing (applying sound correspondences with CopperMT)
def prepare_for_CopperMT(
    data_f,
    out_dir,
    hr_lang,
    lr_lang,
    train_data_dir
):
    global EOS

    # need these to exist for the CopperMT scripts
    for n in ["train", "fine_tune"]:
        for f in os.listdir(train_data_dir):
            f_path = os.path.join(train_data_dir, f)
            if f.startswith(n):
                print("copying", f_path)
                print("\tto", out_dir)
                shutil.copy(f_path, out_dir)

    with open(data_f) as inf:
        data = [line.strip() for line in inf.readlines()]

    if hr_lang in spacy_nlp:
        print(f"Using spacy_tokenize, lang={hr_lang}")
        word_tokenize = spacy_tokenize
    else:
        assert hr_lang in nltk_tokenize_langs
        print(f"Using nltk_tokenize, lang={hr_lang}")
        word_tokenize = nltk_tokenize

    # print("Tokenizing")
    # tokens = []
    # for line in tqdm(data):
    #     line_tokens = line.split()
    #     line_tokens.append(EOS) # add end of sentence indicator
    #     tokens += line_tokens
    # tokens_out_f = os.path.join(out_dir, f"tokens_{hr_lang}_{lr_lang}.{hr_lang}")
    # print("writing tokens to", tokens_out_f)
    # with open(tokens_out_f, "w") as outf:
    #     outf.write("\n".join(tokens) + "\n")

    print("Formatting")
    just_words = []
    for line in tqdm(data):
        words = word_tokenize(line, lang=hr_lang)
        words = [clean_word(w).strip() for w in words]
        just_words += [
            " ".join(w) for w in words 
            if w not in punctuation and w != ""]

    just_words = sorted(list(set(just_words)))
    just_words_out_f = os.path.join(out_dir, f"test_{hr_lang}_{lr_lang}.{hr_lang}")
    print("writing only words to", just_words_out_f)
    with open(just_words_out_f, "w") as outf:
        outf.write("\n".join(just_words) + "\n")

    just_words_out_f_dummy_tgt = os.path.join(out_dir, f"test_{hr_lang}_{lr_lang}.{lr_lang}")
    make_file(just_words_out_f_dummy_tgt, times=len(just_words))
    

def retrieve(
    data_f,
    CopperMT_results_f,
    final_results_f,
    hr_lang,
    lr_lang,
):
    CopperMT_results = read_CopperMT_Results(CopperMT_results_f)

    with open(data_f) as inf:
        data = [line.strip() for line in inf.readlines()]

    if hr_lang in spacy_nlp:
        print(f"Using spacy_tokenize, lang={hr_lang}")
        word_tokenize = spacy_tokenize
    else:
        assert hr_lang in nltk_tokenize_langs
        print(f"Using nltk_tokenize, lang={hr_lang}")
        word_tokenize = nltk_tokenize

    print("Replacing words")
    new_data = []
    for line in tqdm(data):
        line_words = word_tokenize(line, lang=hr_lang)
        for word in line_words:
            if word in CopperMT_results:
                line = line.replace(word, CopperMT_results[word])
        new_data.append(line)
    
    with open(final_results_f, "w") as outf:
        outf.write("\n".join(new_data) + "\n")


def read_CopperMT_Results(results_f):
    with open(results_f) as inf:
        lines = [line.strip() for line in inf.readlines()]
    data_rows = []
    for line in lines:
        split_line = line.split("|")
        if len(split_line) == 4:
            assert split_line[1].strip() == "INFO"
            continue
        elif line.startswith("Generate test with beam="):
            continue
        else:
            # should be a good line :)
            assert any([
                line.startswith("S-"),
                line.startswith("T-"),
                line.startswith("H-"),
                line.startswith("D-"),
                line.startswith("P-"),
            ])
            data_rows.append(line)
    
    print("Blocking CopperMT")
    data = []
    block = []
    for i, line in tqdm(enumerate(data_rows), len=len(data_rows)):
        block.append(line.strip())
        if i % 5 == 0:
            data.append(tuple(block))
            block = []

    print("Getting Copper MT results")
    results = {}
    for S, T, H, D, P in tqdm(data):
        assert S.startswith("S-")
        assert T.startswith("T-")
        assert H.startswith("H-")
        assert D.startswith("D-")
        assert P.startswith("P-")

        S = S.split("\t")[-1].strip()
        T = T.split("\t")[-1].strip()
        H = H.split("\t")[-1].strip()
        D = D.split("\t")[-1].strip()

        assert H == D

        source = "".join(S.split())
        hyp = "".join(H.split())
        assert source not in results
        results[source] = hyp
    return results

def clean_word(word):
    while word[0] in punctuation and len(word) > 0:
        word = word[1:]
    while word[-1] in punctuation and len(word) > 0:
        word = word[:-1]
    return word

def make_file(f, times=1):
    with open(f, "w") as outf:
        for i in range(times):
            outf.write("-\n")

def spacy_tokenize(line, lang):
    global spacy_nlp
    # print(f"spacy_tokenize lang={lang}")
    doc = spacy_nlp[lang](line)
    tokens = [tok.text for tok in doc]
    return tokens

def nltk_tokenize(line, lang):
    # print(f"nltk_tokenize lang={lang}")
    global word_tokenize_langs
    wtlang = word_tokenize_langs[lang]
    tokens = word_tokenize(line.strip(), language=wtlang)
    return tokens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("-hr", "--hr_lang")
    parser.add_argument("-lr", "--lr_lang")
    parser.add_argument("--training_data")
    parser.add_argument("-R", "--CopperMT_results", default="/home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_checkpoint_best_es_an.an/generate-test.txt")
    parser.add_argument("-F", "--function", choices=["prepare", "retrieve"], default="prepare")
    args = parser.parse_args()
    print("arguments")
    for k, v in vars(args).items():
        print(f"{k}: '{v}'")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("hr_CopperMT.py\n")
    args = get_args()
    # prepare for CopperMT
    if args.function == "prepare":
        print("RUNNING 'prepare'")
        prepare_for_CopperMT(
            data_f=args.data,
            out_dir=args.out,
            hr_lang=args.hr_lang,
            lr_lang=args.lr_lang,
            train_data_dir=args.training_data
        )
    # retrieve CopperMT results and postprocess
    elif args.function == "retrieve":
        print("RUNNING 'retrieve'")
        retrieve(
            data_f=args.data,
            results_f=args.CopperMT_results,
            final_results_f=args.out,
            hr_lang=args.hr_lang,
            lr_lang=args.lr_lang,
        )


