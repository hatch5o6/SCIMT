import argparse
from tqdm import tqdm
import json
import csv
import torch
from torchmetrics.regression import JensenShannonDivergence
from scipy.spatial.distance import jensenshannon
from collections import Counter
JSD = JensenShannonDivergence()

from torch.utils.data import DataLoader
from parallel_datasets import MultilingualDataset
from parallel_datasets_sc import SCAlignedMultilingualDataset
from spm_tokenizers import SPMTokenizer
from sc_aligned_spm_tokenizers import SCAlignedSPMTokenizer

from token_overlap_OLD import calc_JSD, calc_jaccard_overlap

def calc_overlap(
    pl,
    cl,
    tl,

    data_csv,
    sc_csv,
    sc_model_id,

    og_spm,
    sc_spm,
    aligned_vocab,

    # spm1,
    # spm1_sc,
    # spm1_type,

    # spm2,
    # spm2_sc,
    # spm2_type,

    VOCAB_SIZE_CAP,
    out_f
):
    ## PRINTING ARGS ##
    print("--------------CALC OVERLAP--------------")
    print(f"pl: `{pl}`")
    print(f"cl: `{cl}`")
    print(f"tl: `{tl}`")

    print(f"data_csv: `{data_csv}`")
    print(f"sc_csv: `{sc_csv}`")
    print(f"sc_model_id: `{sc_model_id}`")

    print(f"og_spm: `{og_spm}`")
    print(f"sc_spm: `{sc_spm}`")
    print(f"aligned_vocab: `{aligned_vocab}`")

    # print(f"spm1: `{spm1}`")
    # print(f"spm1_sc: `{spm1_sc}`")
    # print(f"spm1_type: `{spm1_type}`")

    # print(f"spm2: `{spm2}`")
    # print(f"spm2_sc: `{spm2_sc}`")
    # print(f"spm2_type: `{spm2_type}`")

    print(f"VOCAB_SIZE_CAP: `{VOCAB_SIZE_CAP}`")
    print(f"out_f: `{out_f}`")
    print("----------------------------------------")
    #####

    # assert spm1_type in ["spm", "sc_aligned"]
    # assert spm2_type in ["spm", "sc_aligned"]

    print("CSV ASSERTS")
    csv_asserts(data_csv, pl, cl, tl, IS_SC_FILE=False)
    csv_asserts(sc_csv, pl, cl, tl, IS_SC_FILE=True)

    print("GETTING DATA")
    plain_dataloader = get_sc_dataloader(
        data_csv=data_csv, 
        shuffle=True
    )
    plain_data = get_data_by_lang_pair(plain_dataloader)
    sc_dataloader = get_sc_dataloader(
        data_csv=data_csv,
        sc_data_csv=sc_csv,
        sc_model_id=sc_model_id,
        shuffle=True
    )
    sc_data = get_data_by_lang_pair(sc_dataloader)

    # TESTS --------------------------
    print("TESTING DATA")
    assert sc_data["<TOTAL>"] == plain_data["<TOTAL>"] # asserts same number of sentences in both
    assert list(sc_data.keys()) == list(plain_data.keys()) # assert same lang pairs in both
    for lang_pair, sc_lines in sc_data.items():
        if lang_pair == "<TOTAL>": continue

        plain_lines = plain_data[lang_pair]
        
        assert len(sc_lines) == len(plain_lines)
        for i in range(len(sc_lines)):
            sc_src_tag,     sc_src_line,    sc_sc_line,     sc_tgt_tag,     sc_tgt_line     = sc_lines[i]
            plain_src_tag,  plain_src_line, plain_sc_line,  plain_tgt_tag,  plain_tgt_line  = plain_lines[i]
            assert sc_src_tag == plain_src_tag == lang_pair[0]
            assert sc_src_tag in [pl, cl]
            assert sc_src_line == plain_src_line
            assert plain_sc_line == "<NONE>"
            assert sc_tgt_tag == plain_tgt_tag == tl == lang_pair[1]
            assert sc_tgt_line == plain_tgt_line
    # --------------------------------
    
    print("SORTING DATA BY LANG")
    pl_data = []
    cl_data = []
    pl2cl_data = []
    pl2cl_onlysc_data = []
    tl_data = [] # all tl lines
    tl_pl_data = [] # tl lines aligned with pl
    tl_cl_data = [] # tl lines aligned with cl
    for lang_pair, sc_lines in sc_data.items():
        if lang_pair == "<TOTAL>": continue

        for src_tag, src_line, sc_line, tgt_tag, tgt_line in sc_lines:
            assert tgt_tag == tl
            tl_data.append((tgt_line, tl))

            assert src_tag in [pl, cl]
            if src_tag == pl:
                pl_data.append((src_line, pl))
                assert sc_line != "<NONE>"
                pl2cl_data.append((src_line, sc_line, pl))
                pl2cl_onlysc_data.append((sc_line, pl))
                tl_pl_data.append((tgt_line, tl))
            elif src_tag == cl:
                cl_data.append((src_line, cl))
                assert sc_line == "<NONE>"
                tl_cl_data.append((tgt_line, tl))
    
    assert len(pl_data) == len(pl2cl_data) == len(pl2cl_onlysc_data)
    for x in range(len(pl_data)):
        assert pl_data[x][0] == pl2cl_data[x][0] 
        assert pl2cl_data[x][1] == pl2cl_onlysc_data[x][0]
        assert pl_data[x][1] == pl2cl_data[x][2] == pl2cl_onlysc_data[x][1] == pl
    assert len(tl_pl_data) == len(pl_data)
    assert len(tl_cl_data) == len(cl_data)
    for cl_line, cl_lang in cl_data:
        assert cl_lang == cl
    for tl_line, tl_lang in tl_data + tl_pl_data + tl_cl_data:
        assert tl_lang == tl

    cl_aligned_data = [(t, None, l) for t, l in cl_data]
    tl_aligned_data = [(t, None, l) for t, l in tl_data]
    tl_pl_aligned_data = [(t, None, l) for t, l in tl_pl_data]
    tl_cl_aligned_data = [(t, None, l) for t, l in tl_cl_data]

    print("\nINITIALIZING TOKENIZERS")
    og_tokenizer = SPMTokenizer(spm_name=og_spm, lang_toks=[f"<{pl}>", f"<{cl}>", f"<{tl}>"])
    sc_tokenizer = SPMTokenizer(spm_name=sc_spm, lang_toks=[f"<{pl}>", f"<{cl}>", f"<{tl}>"])
    print("INITIALIZING SCAlignedSPMTokenizer")
    aligned_tokenizer = SCAlignedSPMTokenizer(
        # fr_spm_name=og_spm,
        pl=f"<{pl}>",
        cl=f"<{cl}>",
        tl=f"<{tl}>",
        sc_spm_name=sc_spm,
        aligned_vocab=aligned_vocab,
        lang_toks=[f"<{pl}>", f"<{cl}>", f"<{tl}>"],
        VOCAB_SIZE_CAP=VOCAB_SIZE_CAP
    )

    print(f"\nTOKENIZING TESTS FOR LANGS: \n\tPL: {pl}\n\tCL: {cl}\n\tTL: {tl}")
    PAD = True
    ADD_LANG_TOK = True
    ADD_SPECIAL = True
    # UNCOMMENT THESE TWO BLOCKS AFTER YOU'RE DONE TESTING aligned_tokenizer
    print("og vocab special toks")
    for idx in range(10):
        print(idx, og_tokenizer.idx2token[idx])
    pl_og_batch_toks = og_tokenizer.batch_tokenize(pl_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)
    cl_og_batch_toks = og_tokenizer.batch_tokenize(cl_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)

    print("sc vocab special toks")
    for idx in range(10):
        print(idx, sc_tokenizer.idx2token[idx])
    pl2cl_sc_batch_toks = sc_tokenizer.batch_tokenize(pl2cl_onlysc_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)
    cl_sc_batch_toks = sc_tokenizer.batch_tokenize(cl_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)

    print("aligned vocab special toks")
    for idx in range(10):
        print(idx, aligned_tokenizer.idx2token[idx])
    pl_aligned_batch_toks = aligned_tokenizer.batch_tokenize(pl2cl_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)
    cl_aligned_batch_toks = aligned_tokenizer.batch_tokenize(cl_aligned_data, pad_batch=PAD, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)

    # print("BATCH --------")
    # for seq in cl_aligned_batch_toks:
    #     print(seq, "len:", len(seq))
    # print("NORMAL -------")
    # for t, sct, l in cl_aligned_data[:3]:
    #     idx_seq, spm_seq = aligned_tokenizer.tokenize(t, sct, l, add_lang_token=ADD_LANG_TOK, add_special=ADD_SPECIAL)
    #     print(idx_seq)
    #     print(spm_seq)
    #     print("-")

    # Baseline
    print("\n\nBASELINE")
    og_JSD = calc_JSD(pl_og_batch_toks, cl_og_batch_toks)
    og_jaccard = calc_jaccard_overlap(pl_og_batch_toks, cl_og_batch_toks)
    pl_og_batch_toks_unks = count_unk(pl_og_batch_toks, tag="pl", unk=og_tokenizer.token2idx[og_tokenizer.unk])
    cl_og_batch_toks_unks = count_unk(cl_og_batch_toks, tag="cl", unk=og_tokenizer.token2idx[og_tokenizer.unk])


    # SC
    print("\n\nSC")
    sc_JSD = calc_JSD(pl2cl_sc_batch_toks, cl_sc_batch_toks)
    sc_jaccard = calc_jaccard_overlap(pl2cl_sc_batch_toks, cl_sc_batch_toks)
    pl2cl_sc_batch_toks_unks = count_unk(pl2cl_sc_batch_toks, tag="pl", unk=sc_tokenizer.token2idx[sc_tokenizer.unk])
    cl_sc_batch_toks_unks = count_unk(cl_sc_batch_toks, tag="cl", unk=sc_tokenizer.token2idx[sc_tokenizer.unk])


    # SC-Aligned
    print("\n\nSC-Aligned")
    langx = aligned_tokenizer.lang_toks[0]
    unk_id = aligned_tokenizer.token2idx[aligned_tokenizer.unk][langx]
    print(f"aliged_tokenizer unk = {aligned_tokenizer.unk}, {unk_id}")
    for lx in aligned_tokenizer.lang_toks:
        lx_unk_id = aligned_tokenizer.token2idx[aligned_tokenizer.unk][lx]
        print("lang {lx}, {aligned_tokenizer.unk} = {lx_unk_id}")
        assert lx_unk_id == unk_id

    aligned_JSD = calc_JSD(pl_aligned_batch_toks, cl_aligned_batch_toks)
    aligned_jaccard = calc_jaccard_overlap(pl_aligned_batch_toks, cl_aligned_batch_toks)
    pl_aligned_batch_toks_unks = count_unk(pl_aligned_batch_toks, tag="pl", unk=unk_id)
    cl_aligned_batch_toks_unks = count_unk(cl_aligned_batch_toks, tag="cl", unk=unk_id)


    with open(out_f, "w") as outf:
        outf.write("BASELINE\n")
        outf.write("JSD\n")
        outf.write(json.dumps(og_JSD) + "\n")
        outf.write("Jaccard")
        outf.write(json.dumps(og_jaccard) + "\n")
        outf.write(f"pl unk: {json.dumps(pl_og_batch_toks_unks, indent=2)}\n")
        outf.write(f"cl unk: {json.dumps(cl_og_batch_toks_unks, indent=2)}\n")

        outf.write("\n\nSC")
        outf.write("JSD\n")
        outf.write(json.dumps(sc_JSD) + "\n")
        outf.write("Jaccard")
        outf.write(json.dumps(sc_jaccard) + "\n")
        outf.write(f"pl unk: {json.dumps(pl2cl_sc_batch_toks_unks, indent=2)}\n")
        outf.write(f"cl unk: {json.dumps(cl_sc_batch_toks_unks, indent=2)}\n")

        outf.write("\n\nSC-Aligned")
        outf.write("JSD\n")
        outf.write(json.dumps(aligned_JSD) + "\n")
        outf.write("Jaccard")
        outf.write(json.dumps(aligned_jaccard) + "\n")
        outf.write(f"pl unk: {json.dumps(pl_aligned_batch_toks_unks, indent=2)}\n")
        outf.write(f"cl unk: {json.dumps(cl_aligned_batch_toks_unks, indent=2)}\n")

def count_unk(token_seqs, unk=0, tag="pl"):
    print(f"count_unk unk={unk}")
    total_tokens = 0
    unk_tokens = 0
    seqs_with_unk = 0
    for seq in token_seqs:
        SEQ_HAS_UNK = False
        for tok in seq:
            total_tokens += 1
            if tok == unk:
                unk_tokens +=1
                SEQ_HAS_UNK = True
        if SEQ_HAS_UNK:
            seqs_with_unk += 1

    score = {
        f"{tag} unk_tokens": unk_tokens, 
        "total_tokens": total_tokens,
        "total_seqs": len(token_seqs),
        "unk/total_tokens": unk_tokens / total_tokens,
        "unk_tokens/seq": unk_tokens / len(token_seqs),
        "seqs_with_unk": seqs_with_unk,
        "seqs_with_unk/total_seqs": seqs_with_unk / len(token_seqs)
    }
    print("UNK TOKENS", score)
    return score

def get_data_by_lang_pair(dataloader):
    pairs = {"<TOTAL>": 0}
    for src_tags, src_lines, sc_lines, tgt_tags, tgt_lines in dataloader:
        batch = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
        for src_tag, src_line, sc_line, tgt_tag, tgt_line in batch:
            lang_pair = (src_tag, tgt_tag)
            if lang_pair not in pairs:
                pairs[lang_pair] = []
            pairs[lang_pair].append((src_tag, src_line, sc_line, tgt_tag, tgt_line))
            pairs["<TOTAL>"] += 1
    return pairs


def get_sc_dataloader(
    data_csv,
    sc_data_csv=None,
    sc_model_id=None,
    upsample=False,
    shuffle=False,
    PRINT:int=None
):
    assert isinstance(PRINT, int) or PRINT == None
    print("SC DATA:")
    # SHOULD ONLY DO THIS ON TRAINING PROBABLY
    # Upsample=False and shuffle=True used for training data in pretrain -> finetune scenarios.
    sc_dataset = SCAlignedMultilingualDataset(
        data_csv=data_csv,
        sc_data_csv=sc_data_csv,
        sc_model_id=sc_model_id,
        append_src_lang_tok=False,
        append_tgt_lang_tok=False,
        append_tgt_to_src=False,
        upsample=upsample,
        shuffle=shuffle
    )
    sc_dataloader = DataLoader(
        sc_dataset,
        batch_size=1024,
        shuffle=False
    )

    if isinstance(PRINT, int):
        i = 0
        for src_tags, src_lines, sc_lines, tgt_tags, tgt_lines in sc_dataloader:
            batch = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
            for src_tag, src_line, sc_line, tgt_tag, tgt_line in batch:
                print(f"--------------- {i} -----------------")
                print(f"(src) {src_tag}: `{src_line}`")
                print(f"(sc)  {src_tag}: `{sc_line}`")
                print(f"(tgt) {tgt_tag}: `{tgt_line}`")
                i += 1
            
            if i >= PRINT:
                break
        return None
    else:
        return sc_dataloader

def csv_asserts(data_csv, pl, cl, tl, IS_SC_FILE=False):
    print("csv asserting:", data_csv)
    with open(data_csv, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    rows = rows[1:]

    assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]
    for src_lang, tgt_lang, src_path, tgt_path in rows:
        assert tgt_lang == tl
        assert "SC_{SC_MODEL_ID}" not in tgt_path

        assert src_lang in [pl, cl]
        if IS_SC_FILE and src_lang == pl:
            assert "SC_{SC_MODEL_ID}" in src_path
            assert src_path.endswith(f"SC_{{SC_MODEL_ID}}_{pl}2{cl}.txt")
            # assert sc_model_id is not None
        else:
            assert "SC_{SC_MODEL_ID}" not in src_path


# def read_data(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE=False):
#     assert_model_id(data_csv, pl, cl, tl, sc_model_id, IS_SC_FILE)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl", help="parent language")
    parser.add_argument("--cl", help="child language")
    parser.add_argument("--tl", help="target language")

    parser.add_argument("--data_csv") # should be augmented traininng csv, and only have two lang pairs (e.g. mfe-en, fr-en)
    parser.add_argument("--sc_csv") # should be sc version of data_csv
    parser.add_argument("--sc_model_id")

    # # for first of lang pairs in data_csv
    # parser.add_argument("--spm1")
    # parser.add_argument("--spm1_sc")
    # parser.add_argument("--spm1_type", choices=["spm", "sc_aligned"])

    # # for second of lang pairs in data_csv
    # parser.add_argument("--spm2")
    # parser.add_argument("--spm2_sc")
    # parser.add_argument("--spm2_type", choices=["spm", "sc_aligned"])

    parser.add_argument("--og_spm", help="tokenizer trained on original data. Should be trained on pl, cl, and tl.")
    parser.add_argument("--sc_spm", help="tokenizer trained on sc-converted (e.g. SC_fr2mfe) data. Should be trained on pl, cl, and tl.")
    parser.add_argument("--aligned_vocab", help="vocab created by alignment (align_tokens.py, sc_aligned_spm_tokenizers.py) with sc_spm. Should be trained on pl, cl, tl.")


    # parser.add_argument("--is_parallel", action="store_true")
    parser.add_argument("--VOCAB_SIZE_CAP", type=int, default=32000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t--{k} = '{v}'")
    return args

if __name__ == "__main__":
    print("----------------------------------")
    print("###### token_overlap_csv.py ######")
    print("----------------------------------")
    args = get_args()
    calc_overlap(
        pl=args.pl,
        cl=args.cl,
        tl=args.tl,

        data_csv=args.data_csv,
        sc_csv=args.sc_csv,
        sc_model_id=args.sc_model_id,

        og_spm=args.og_spm,
        sc_spm=args.sc_spm,
        aligned_vocab=args.aligned_vocab,

        # spm1=args.spm1,
        # spm1_sc=args.spm1_sc,
        # spm1_type=args.spm1_type,

        # spm2=args.spm2,
        # spm2_sc=args.spm2_sc,
        # spm2_type=args.spm2_type,

        VOCAB_SIZE_CAP=args.VOCAB_SIZE_CAP,
        out_f=args.out
    )
    