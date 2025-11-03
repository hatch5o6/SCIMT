import argparse
import os
import shutil
from tqdm import tqdm
from string import punctuation
punctuation += "-—¡¿؟؛،٪»«›‹”“〞❮❯❛❟%."
import re
import csv

from indicnlp.tokenize import indic_tokenize 
indicnlp_langs = {"hi", "as", "bn", "bho"}

from camel_tools.tokenizers.word import simple_word_tokenize as camel_simple_word_tokenize
arabic_langs = {"ar", "aeb", "apc"}

import nltk
from nltk.tokenize import word_tokenize as NLTK_word_tokenize
nltk_tokenize_langs = {
    "de": "german",
    "hsb": "czech",
    "cs": "czech",
    "en": "english",
    "djk": "english",
    "NGfr": "french",
    "NGmfe": "french",
    "fr": "french",
    "mfe": "french",
    
    # fake langs for testing:
    "bren": "french",
    "dan": "french",
    "tho": "czech",
    "mas": "czech"
}

import spacy
es_nlp = spacy.load('es_core_news_sm', exclude=["tagger", "parser", "ner", "lemmatizer", "textcat", "custom", "entity_linker", "entity_ruler", "textcat_multilabel", "trainable_lemmatizer", "morphologizer", "attribute_ruler", "senter", "sentencizer", "tok2vec", "transformer"])
multi_nlp = spacy.load('xx_sent_ud_sm', exclude=["tagger", "parser", "ner", "lemmatizer", "textcat", "custom", "entity_linker", "entity_ruler", "textcat_multilabel", "trainable_lemmatizer", "morphologizer", "attribute_ruler", "senter", "sentencizer", "tok2vec", "transformer"])
spacy_nlp = {
    "es": es_nlp,
    "an": es_nlp,
    "oc": es_nlp,
    "ast": es_nlp,

    "lua": multi_nlp,
    "bem": multi_nlp,
    "ewe": multi_nlp,
    "fon": multi_nlp
}

from parallel_datasets import MultilingualDataset
from torch.utils.data import DataLoader

# EOS = "<EOS>"

# function for preparing HR lang data for preprocessing (applying sound correspondences with CopperMT)
def read_csv(f):
    pass

def prepare_for_CopperMT(
    data_f,
    out_dir,
    hr_lang,
    lr_lang,
    train_data_dir,
    limit_lang=None
):
    # global EOS

    # need these to exist for the CopperMT scripts
    for n in ["train", "fine_tune"]:
        for f in os.listdir(train_data_dir):
            f_path = os.path.join(train_data_dir, f)
            if f.startswith(n):
                print("copying", f_path)
                print("\tto", out_dir)
                shutil.copy(f_path, out_dir)


    if data_f.endswith(".csv"):
        print("prepare_for_CopperMT READING DATASET CSV FILE:", data_f)
        limit_src_langs = None
        limit_tgt_langs = None
        if limit_lang != None:
            limit_src_langs = [limit_lang]
            limit_tgt_langs = [limit_lang]
        
        print("limit_src_langs:", limit_src_langs)
        print("limit_tgt_langs:", limit_tgt_langs)

        dataset_src_filtered = MultilingualDataset(
            data_csv=data_f,
            append_src_lang_tok=True,
            append_tgt_lang_tok=True,
            seed=None,
            upsample=False,
            shuffle=False,
            limit_src_langs=limit_src_langs,
            limit_tgt_langs=None,
            CAN_RETURN_ZERO=True
        )
        src_filtered_data = set()
        dataloader_src_filtered = DataLoader(dataset_src_filtered, batch_size=100)
        for src_segs, tgt_segs in dataloader_src_filtered:
            batch = list(zip(src_segs, tgt_segs))
            for src_seg, tgt_seg in batch:
                if not src_seg.startswith(f"<{limit_lang}>"):
                    print("SRC SEG DOES NOT START WITH CORRECT LANG TOKEN")
                    print("\tlimit_lang:", type(limit_lang), limit_lang)
                    print("\tSRC_SEG:", src_seg)
                assert src_seg.startswith(f"<{limit_lang}>")
                src_seg = src_seg[len(f"<{limit_lang}>"):].strip()
                src_filtered_data.add(src_seg)
        print(f"\t{len(src_filtered_data)} SRC-FILTERED SEGS")

        dataset_tgt_filtered = MultilingualDataset(
            data_csv=data_f,
            append_src_lang_tok=True,
            append_tgt_lang_tok=True,
            seed=None,
            upsample=False,
            shuffle=False,
            limit_src_langs=None,
            limit_tgt_langs=limit_tgt_langs,
            CAN_RETURN_ZERO=True
        )
        tgt_filtered_data = set()
        dataloader_tgt_filtered = DataLoader(dataset_tgt_filtered, batch_size=100)
        for src_segs, tgt_segs in dataloader_tgt_filtered:
            batch = list(zip(src_segs, tgt_segs))
            for src_seg, tgt_seg in batch:
                if not tgt_seg.startswith(f"<{limit_lang}>"):
                    print("TGT SEG DOES NOT START WITH CORRECT LANG TOKEN")
                    print("\tlimit_lang:", type(limit_lang), limit_lang)
                    print("\tTGT_SEG:", tgt_seg)
                assert tgt_seg.startswith(f"<{limit_lang}>")
                tgt_seg = tgt_seg[len(f"<{limit_lang}>"):].strip()
                tgt_filtered_data.add(tgt_seg)
        print(f"\t{len(tgt_filtered_data)} TGT-FILTERED SEGS")

        data = list(src_filtered_data.union(tgt_filtered_data))
        print(f"\t{len(data)} UNIQUE SEGS")
    else:
        with open(data_f) as inf:
            data = [line.strip() for line in inf.readlines()]

    if hr_lang in spacy_nlp:
        print(f"Using spacy_tokenize, lang={hr_lang}")
        word_tokenize = spacy_tokenize
    elif hr_lang in indicnlp_langs:
        print(f"Using indic_word_tokenize, lang={hr_lang}")
        word_tokenize = indic_word_tokenize
    elif hr_lang in arabic_langs:
        print(f"Using camel_tokenize, lang={hr_lang}")
        word_tokenize = camel_tokenize
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
        tokens = line.split()
        for token in tokens:
            words = word_tokenize(token, lang=hr_lang)
            words = [clean_word(w) for w in words]
            just_words += [
            " ".join(w) for w in words 
            if w not in punctuation and w != ""]

        # words = word_tokenize(line, lang=hr_lang)
        # words = [clean_word(w) for w in words]
        # just_words += [
        #     " ".join(w) for w in words 
        #     if w not in punctuation and w != ""]

    just_words = sorted(list(set(just_words)))
    just_words_out_f = os.path.join(out_dir, f"test_{hr_lang}_{lr_lang}.{hr_lang}")
    print("writing only words to", just_words_out_f)
    with open(just_words_out_f, "w") as outf:
        outf.write("\n".join(just_words) + "\n")

    just_words_out_f_dummy_tgt = os.path.join(out_dir, f"test_{hr_lang}_{lr_lang}.{lr_lang}")
    make_file(just_words_out_f_dummy_tgt, times=len(just_words))
    
def get_data_fs_from_csv(csv_f, lang):
    with open(csv_f, newline="") as inf:
        rows = [row for row in csv.reader(inf)]
    header = rows[0]
    rows = [tuple(row) for row in rows[1:]]
    data_fs = set()
    for src_lang, tgt_lang, src_f, tgt_f in rows:
        if src_lang == lang:
            assert tgt_lang != lang
            data_fs.add(src_f)
        elif tgt_lang == lang:
            assert src_lang != lang
            data_fs.add(tgt_f)
    return list(data_fs)

def retrieve(
    data_f,
    CopperMT_results_f,
    CopperMT_SMT_results_src_f,
    CopperMT_SMT_results_tgt_f,
    final_results_f,
    hr_lang,
    lr_lang,
    MODEL_ID=""
):
    NOT_IN_COPPER_MT_RESULTS = set()

    if CopperMT_results_f is not None:
        CopperMT_results = read_CopperMT_Results(CopperMT_results_f)
        import json
        f = "/home/hatch5o6/Cognate/code/NMT/CopperMT_RNN_RESULTS.json"
        print("Writing to", f)
        with open(f, "w") as outf:
            outf.write(json.dumps(CopperMT_results, ensure_ascii=False, indent=2))
    else:
        assert CopperMT_SMT_results_src_f is not None
        assert CopperMT_SMT_results_tgt_f is not None
        CopperMT_results = read_CopperMT_SMT_RESULTS(
            results_src=CopperMT_SMT_results_src_f,
            results_hyp=CopperMT_SMT_results_tgt_f
        )
        import json
        f = "/home/hatch5o6/Cognate/code/NMT/CopperMT_SMT_RESULTS.json"
        print("Writing to", f)
        with open(f, "w") as outf:
            outf.write(json.dumps(CopperMT_results, ensure_ascii=False, indent=2))

    USE_FINAL_RESULTS_F = True
    if data_f.endswith(".csv"):
        USE_FINAL_RESULTS_F = False
        print("retrieve READING DATASET CSV FILE:", data_f)
        data_fs = get_data_fs_from_csv(data_f, lang=hr_lang)
    else:
        assert final_results_f != None
        data_fs = [data_f]
    
    for data_f in data_fs:
        print(f"-- RETRIEVING RESULTS FROM {data_f} --")
        if USE_FINAL_RESULTS_F:
            assert len(data_fs) == 1
            write_out_f = final_results_f
        else:
            ext = data_f.split(".")[-1]
            write_out_f = f"{data_f[:-len(ext)]}SC_{MODEL_ID}_{hr_lang}2{lr_lang}.{ext}"
        print("\twill write output to", write_out_f)

        with open(data_f) as inf:
            data = [line.strip() for line in inf.readlines()]

        print("HR_LANG", hr_lang)
        if hr_lang in spacy_nlp:
            print(f"Using spacy_tokenize, lang={hr_lang}")
            word_tokenize = spacy_tokenize
        elif hr_lang in indicnlp_langs:
            print(f"Using indic_word_tokenize, lang={hr_lang}")
            word_tokenize = indic_word_tokenize
        elif hr_lang in arabic_langs:
            print(f"Using camel_tokenize, lang={hr_lang}")
            word_tokenize = camel_tokenize
        else:
            assert hr_lang in nltk_tokenize_langs
            print(f"Using nltk_tokenize, lang={hr_lang}")
            word_tokenize = nltk_tokenize

        print("Replacing words")
        new_data = []
        unique_orig_words = set()
        for line in tqdm(data):
            tokens = line.split()
            for t, token in enumerate(tokens):
                token_words = word_tokenize(token, lang=hr_lang)
                for w, word in enumerate(token_words):
                    cleaned_word = clean_word(word)
                    unique_orig_words.add(cleaned_word)
                    assert cleaned_word in word
                    if cleaned_word in CopperMT_results:
                        word_result = CopperMT_results[cleaned_word]
                        assert isinstance(word_result, str)
                        # We are not converting unknown char to <unk> and then retrieving results (which might be a list rather than a str)
                        # Words with an unknown char we'll just skip. The predictions look noisy. I think we might be better off just keeping words with unk chars as is in the original data.
                        word = word.replace(cleaned_word, word_result)
                    else:
                        NOT_IN_COPPER_MT_RESULTS.add(cleaned_word)
                    token_words[w] = word
                token = "".join(token_words)
                tokens[t] = token
            line = " ".join(tokens)

            # line_words = word_tokenize(line, lang=hr_lang)
            # for w, word in enumerate(line_words):
            #     cleaned_word = clean_word(word)
            #     assert cleaned_word in word
            #     if cleaned_word in CopperMT_results:
            #         word = word.replace(cleaned_word, CopperMT_results[cleaned_word])
            #     else:
            #         NOT_IN_COPPER_MT_RESULTS.add(cleaned_word)
            #     line_words[w] = word.strip()
            # line = " ".join(line_words).strip()
                
            new_data.append(line)

        print("NOT IN COPPER MT RESULTS", len(NOT_IN_COPPER_MT_RESULTS))
        for item in NOT_IN_COPPER_MT_RESULTS:
            print(f"\t- `{item}`")
        
        print("Writing final_results to", write_out_f)
        with open(write_out_f, "w") as outf:
            outf.write("\n".join(new_data) + "\n")

        print("UNIQUE WORDS IN ORIG DATA:", len(unique_orig_words))
        print("WORDS IN COPPER MT RESULTS:", len(CopperMT_results))
    
def get_test_results(source_f, results_f, out_f):
    with open(source_f) as inf:
        source = [line.strip() for line in inf.readlines()]
    results = read_CopperMT_Results(results_f, RETURN_SPACED=True)
    hyps = []
    NOT_IN_RESULTS = []
    for src_word in source:
        # print("SRC WORD:", src_word)
        # assert src_word in results.keys()
        if src_word not in results.keys():
            print(f"src_word `{src_word}` not in results. Will try replacing _ with <unk>: ")
            src_word = src_word.replace("_", "<unk>")
            print(f"\tFixed src_word: `{src_word}`")
            # assert src_word in results.keys()

        if src_word in results.keys():
            hyp = results[src_word].strip()
        else:
            print(f"\t\tsrc_word `{src_word}` still not in results :(. Will use src_word as hypothesis.")
            NOT_IN_RESULTS.append(src_word)
            hyp = src_word.strip()
        assert isinstance(hyp, str)
        hyps.append(hyp)
    with open(out_f, "w") as outf:
        outf.write("\n".join(hyps) + "\n")
    
    print("\n\nCOULD NOT FIND HYPOTHESES FOR THESE SRC_WORDS. WE USED THE SRC_WORD AS THE HYPOTHESIS.")
    for item in NOT_IN_RESULTS:
        print(f"\t-`{item}`")

def read_smt_result(f):
    with open(f) as inf:
        data = [
            re.sub(r'\s+', '', line.strip())
            for line in inf.readlines()
        ]
    return data

def read_CopperMT_SMT_RESULTS(results_src, results_hyp):
    results_src = read_smt_result(results_src)
    results_hyp = read_smt_result(results_hyp)
    assert len(results_src) == len(results_hyp)
    results_list = list(zip(results_src, results_hyp))

    results = {}
    for src_seg, tgt_seg in results_list:
        if src_seg not in results:
            results[src_seg] = tgt_seg
        else:
            assert results[src_seg] == tgt_seg
    return results

def read_CopperMT_Results(results_f, RETURN_SPACED=False):
    with open(results_f) as inf:
        lines = [line.strip() for line in inf.readlines()]
    
    # print("\nLAST 10 LINES IN LINES")
    # for line in lines[-10:]:
    #     print(line)
    # print("\n\n")

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
    
    # print("\nLAST 10 LINES IN DATA ROWS")
    # for line in data_rows[-10:]:
    #     print(line)
    # print("\n\n")

    print("Blocking CopperMT")
    data = []
    block = []
    for i, line in tqdm(enumerate(data_rows), total=len(data_rows)):
        if i > 0 and i % 5 == 0:
            data.append(tuple(block))
            block = []
        block.append(line.strip())
    if len(block) > 0:
        assert len(block) == 5
        data.append(tuple(block))

    # print("\nLAST 3 BLOCKS IN DATA")
    # for block in data[-3:]:
    #     print(block)
    # print("\n\n")

    # print("Data")
    # for i in data[:5]:
    #     print(type(i), i)

    print("Getting Copper MT results")
    # results = {'<unk>': []}
    results = {}
    # results_list = []
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

        if RETURN_SPACED:
            source = S
            hyp = H
        else:
            source = "".join(S.split())
            hyp = "".join(H.split())
        # results_list.append((source, hyp))
        if source in results:
            print("SOURCE IN RESULTS")
            print("source:", source)
            print("hyp:", hyp)
            print(f"results['{source}']:", results[source])
        if "<unk>" not in source:
            assert source not in results
            results[source] = hyp
        else:
            if source not in results:
                results[source] = hyp
            else:
                # print(f"TRYING TO SET results['{source}'] (='{results[source]}') to {hyp}")
                # assert results[source] == hyp
                if isinstance(results[source], str):
                    results[source] = [results[source]]
                assert isinstance(results[source], list)
                if hyp not in results[source]:
                    results[source].append(hyp)

    # if RETURN_LIST:
    #     return results_list
    # else:
    return results

def clean_word(word):
    while len(word) > 0 and word[0] in punctuation:
        word = word[1:]
    while len(word) > 0 and word[-1] in punctuation:
        word = word[:-1]
    return word.strip()

def make_file(f, times=1):
    with open(f, "w") as outf:
        for i in range(times):
            outf.write("-\n")

def spacy_tokenize(line, lang):
    global spacy_nlp
    # print(f"spacy_tokenize lang={lang}")
    doc = spacy_nlp[lang](line)
    tokens = [tok.text for tok in doc]
    final_tokens = []
    for tok in tokens:
        subtoks = tok.split()
        for subtok in subtoks:
            final_tokens.append(subtok.strip())
    return final_tokens

def nltk_tokenize(line, lang):
    # print(f"nltk_tokenize lang={lang}")
    global nltk_tokenize_langs
    wtlang = nltk_tokenize_langs[lang]
    tokens = NLTK_word_tokenize(line.strip(), language=wtlang)
    final_tokens = []
    for tok in tokens:
        subtoks = tok.split()
        for subtok in subtoks:
            final_tokens.append(subtok.strip())
    return final_tokens

def indic_word_tokenize(line, lang):
    global indicnlp_langs
    assert lang in indicnlp_langs
    tokens = indic_tokenize.trivial_tokenize_indic(line)
    final_tokens = []
    for tok in tokens:
        subtoks = tok.split()
        for subtok in subtoks:
            final_tokens.append(subtok.strip())
    return final_tokens

def camel_tokenize(line, lang):
    global arabic_langs
    assert lang in arabic_langs
    tokens = camel_simple_word_tokenize(line, split_digits=True)
    final_tokens = []
    for tok in tokens:
        subtoks = tok.split()
        for subtok in subtoks:
            final_tokens.append(subtok.strip())
    return final_tokens



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--test_src")
    parser.add_argument("--out")
    parser.add_argument("-hr", "--hr_lang")
    parser.add_argument("-lr", "--lr_lang")
    parser.add_argument("--training_data")
    parser.add_argument("--limit_lang")
    parser.add_argument("-R", "--CopperMT_results", 
                        # default="/home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_checkpoint_best_es_an.an/generate-test.txt"
    )
    parser.add_argument("-S", "--CopperMT_SMT_results", help="comma-delimited list, must be len 2, src file first, hyp file second (e.g. src_file.txt,hyp_file.txt")
    parser.add_argument("-F", "--function", choices=["prepare", "retrieve", "get_test_results"], default="prepare")
    parser.add_argument("-M", "--MODEL_ID", default="")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: '{v}'")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("############################")
    print("###### hr_CopperMT.py ######")
    print("############################")
    args = get_args()
    # prepare for CopperMT
    if args.function == "prepare":
        print("RUNNING 'prepare'")
        prepare_for_CopperMT(
            data_f=args.data,
            out_dir=args.out,
            hr_lang=args.hr_lang,
            lr_lang=args.lr_lang,
            train_data_dir=args.training_data,
            limit_lang=args.limit_lang
        )
    # retrieve CopperMT results and postprocess
    elif args.function == "retrieve":
        print("RUNNING 'retrieve'")
        CopperMT_SMT_results_fs = None
        CopperMT_SMT_results_src_f, CopperMT_SMT_results_tgt_f = None, None
        if args.CopperMT_SMT_results:
            CopperMT_SMT_results_fs = [d.strip() for d in args.CopperMT_SMT_results.split(",")]
            assert len(CopperMT_SMT_results_fs) == 2
            CopperMT_SMT_results_src_f, CopperMT_SMT_results_tgt_f = tuple(CopperMT_SMT_results_fs)
        retrieve(
            data_f=args.data,
            CopperMT_results_f=args.CopperMT_results,
            CopperMT_SMT_results_src_f=CopperMT_SMT_results_src_f,
            CopperMT_SMT_results_tgt_f=CopperMT_SMT_results_tgt_f,
            final_results_f=args.out,
            hr_lang=args.hr_lang,
            lr_lang=args.lr_lang,
            MODEL_ID=args.MODEL_ID
        )
    elif args.function == "get_test_results":
        print("RUNNING 'get_test_results'")
        get_test_results(
            source_f=args.test_src,
            results_f=args.data,
            out_f=args.out
        )


