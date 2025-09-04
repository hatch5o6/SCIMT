import argparse
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from spm_tokenizers import SPMTokenizer
import Levenshtein as lev
from collections import Counter
import json
from tqdm import tqdm
import os
import shutil

"""
Affiliate this with the spm tokenizer training.
"""

def read_data(f, TOTAL):
    lines = []
    with open(f) as inf:
        if TOTAL != None:
            print(f"READING {TOTAL} LINES FROM {f}")
            while len(lines) < TOTAL:
                lines.append(inf.readline().strip())
        else:
            print("READING ALL LINES FROM {f}")
            lines = [l.strip() for l in inf.readlines()]
    print(f"\tN LINES={len(lines)}")
    return lines

def get_tok_map(tokens):
    # provided a list of tokens, map each char to the token, char index
    # We'll do this with a list of tuples: [(char, tok_idx, char_idx)]
    chars = []
    for t, tok in enumerate(tokens):
        for c, char in enumerate(tok):
            chars.append((char, t, c))
    return chars

def get_ops_dict(ops):
    # for o, (op, x1, x2) in ops:
    #     p = o - 1
    #     pop, px1, px2 = None, None, None
    #     if p > 0:
    #         pop, px1, px2 = ops[p]
    
    #     n = o + 1
    #     nop, nx1, nx2 = None, None, None
    #     if n < len(ops):
    #         nop, nx1, nx2 = ops[n]

    #     if op == "replace":
    #         assert pop in [None, "delete", "insert"]
    #         assert nop in [None, "delete", "insert"]
    #     elif op == "delete":
    #         assert (pop in [None, "replace", "insert"]) or (pop == "delete" and px1 == x1 - 1 and px2 == x2)
    #     elif op == "insert":
    #         assert (nop in [None, "replace", "delete"]) or (pop == "insert" and px1 == x1 and px2 = x2 - 1)

    ops_dict = {}
    for o, (op, x1, x2) in enumerate(ops):
        if x1 not in ops_dict:
            ops_dict[x1] = []
        ops_dict[x1].append((op, x1, x2))
    return ops_dict

def get_tied_char_alignments(fr_toks_tied, sc_toks_tied):
    ops = lev.editops(fr_toks_tied, sc_toks_tied)
    # print("OPS", ops)
    ops_dict = get_ops_dict(ops)
    frx = 0
    scx = 0
    fr2sc_tied_mappings = []
    # print("ITERATING THROUGH OPS DICT")
    while frx < len(fr_toks_tied) or scx < len(sc_toks_tied):
        # Once frx reaches len(fr_toks_tied), it should stop incrementing
        # Once scx reaches len(sc_toks_tied), it should also stop incrementing
        assert frx <= len(fr_toks_tied)
        assert scx <= len(sc_toks_tied)
        # print("----------------------------------------------")
        # print(f"LEN FR TOKS TIED ({len(fr_toks_tied)}), FRX:", frx)
        if frx < len(fr_toks_tied):
            # print("\t", fr_toks_tied[frx])
            pass
        else:
            # if it isn't less than, it must be equal
            assert frx == len(fr_toks_tied)
            # print("\tEND")
        # print(f"LEN SC TOKS TIED ({len(fr_toks_tied)}), SCX:", scx)
        if scx < len(sc_toks_tied):
            # print("\t", sc_toks_tied[scx])
            pass
        else:
            # if it isn't less than, it must be equal
            assert scx == len(sc_toks_tied)
            # print("\tEND")

        if frx in ops_dict:
            TO_POP = frx
            for op, opfrx, opscx in ops_dict[frx]:
                # print(f"\tops_dict[{frx}]", (op, opfrx, opscx))
                assert frx == opfrx
                assert scx == opscx
                if frx < len(fr_toks_tied) and scx == len(sc_toks_tied):
                    assert op == "delete"
                elif frx == len(fr_toks_tied) and scx < len(sc_toks_tied):
                    assert op == "insert"

                if op == "replace":
                    # print(f"\tappending replace (FRX:{frx}, SCX:{scx})")
                    fr2sc_tied_mappings.append((frx, scx))
                    frx += 1
                    scx += 1
                elif op == "delete":
                    # fr2sc_tied_mappings.append((frx, scx - 1))
                    # print(f"\tappending delete (FRX:{frx}, SCX:None)")
                    fr2sc_tied_mappings.append((frx, None))
                    frx += 1
                elif op == "insert":
                    # fr2sc_tied_mappings.append((frx, scx))
                    # print(f"\tappending insert (FRX:None, SCX:{scx})")
                    fr2sc_tied_mappings.append((None, scx))
                    scx += 1
            ops_dict.pop(TO_POP)
        else:
            # print(f"\tappending equal (FRX:{frx}, SCX:{scx})")
            assert fr_toks_tied[frx] == sc_toks_tied[scx]
            fr2sc_tied_mappings.append((frx, scx))
            frx += 1
            scx += 1
    # print("------- END -------")
    # print("LEN FR TOKS TIED:", len(fr_toks_tied))
    # print("FRX:", frx)
    # print("\nLEN SC TOKS TIED:", len(sc_toks_tied))
    # print("SCX:", scx)
    assert frx == len(fr_toks_tied)
    assert scx == len(sc_toks_tied)
    assert len(ops_dict) == 0
    return fr2sc_tied_mappings

def get_fr_toks(
    fr2sc_tied_mappings, 
    fr_toks_tied, 
    sc_tok_map,
    sc_tok_seq
):
    """
    fr_toks_tied == the whole sequence (with _ for ws). The tokens are 'tied' together (concatenated) and we're going to figure out where the token boundaries are. :)
    """
    frx = 0
    scx = 0
    next_mapping = 0
    fr_tok_map = []
    # while frx < len(fr_toks_tied) and scx < len(sc_tok_map):
    while next_mapping < len(fr2sc_tied_mappings):
        assert frx <= len(fr_toks_tied) # should never exceed length
        assert scx <= len(sc_tok_map) # should never exceed length

        if frx < len(fr_toks_tied):
            fr_char = fr_toks_tied[frx]
        else:
            # if frx is len(fr_toks_tied), assert we have insertion
            assert frx == len(fr_toks_tied)
            assert scx < len(sc_tok_map)
            fr_char = None
            assert (None, scx) == fr2sc_tied_mappings[next_mapping]

        if scx < len(sc_tok_map):
            sc_char, sc_tok_idx, sc_char_idx = sc_tok_map[scx]
        else:
            # if scx is len(sc_tok_map), assert we have deletion
            assert scx == len(sc_tok_map)
            assert frx < len(fr_toks_tied)
            sc_char, sc_tok_idx, sc_char_idx = None, None, None
            assert (frx, None) == fr2sc_tied_mappings[next_mapping]

        if (frx, scx) == fr2sc_tied_mappings[next_mapping]:
            fr_tok_map.append((fr_char, sc_char, sc_tok_idx, sc_char_idx))
            frx += 1
            scx += 1
        elif (frx, None) == fr2sc_tied_mappings[next_mapping]:
            fr_tok_map.append((fr_char, None, None, None))
            frx += 1
        elif (None, scx) == fr2sc_tied_mappings[next_mapping]:
            fr_tok_map.append((None, sc_char, sc_tok_idx, sc_char_idx))
            scx += 1
        else:
            return ValueError(f"Current frx: {frx}, current scx: {scx}, mapping: ({fr2sc_tied_mappings[next_mapping]})\n\t-Each next mapping in fr2sc_tied_mappings must be (frx, scx), (frx, None), or (None, scx).\n\t-This mapping should be ({frx}, {scx}), ({frx}, None), or (None, {scx}).")

        next_mapping += 1

    assert frx == len(fr_toks_tied)
    assert scx == len(sc_tok_map)

    fr_toks = []
    fr_tok = ""
    cur_tok_idx = None
    # print("\n\nBUIDLING FR TOKS:")
    for ix, (fr_char, sc_char, sc_tok_idx, sc_char_idx) in enumerate(fr_tok_map):
        # print("----------------------------------------------------------")
        # print(ix, (fr_char, sc_char, sc_tok_idx, sc_char_idx))
        # print("\tFR_TOK:", fr_tok)
        # print("\tCUR_TOK_IDX:", cur_tok_idx)
        # print("\tSC_TOK_IDX:", sc_tok_idx)
        if sc_tok_idx == None:
            assert cur_tok_idx is not None
            sc_tok_idx = cur_tok_idx
            # print("\t\tWILL KEEP SC_TOK_IDX at", cur_tok_idx)
        if ix > 0 and sc_tok_idx != cur_tok_idx:
            # print("\t\tADDING TOK:", fr_tok)
            if sc_tok_idx is not None:
                if fr_tok == "":
                    fr_tok = None
                assert cur_tok_idx is not None
                assert sc_tok_seq[cur_tok_idx] is not None
                fr_toks.append((fr_tok, sc_tok_seq[cur_tok_idx], cur_tok_idx))
            else:
                fr_toks.append((fr_tok, None, None))
            fr_tok = ""
        # Actually I don't think we need this assertion since these are just fr and sc characters that are ALIGNED, which doesn't necessarily mean they're the same, in fact they won't be in a large number of cases.... that's the whole reason we're doing this thing :)
        # assert fr_char == sc_tok_seq[sc_tok_idx][sc_char_idx]
        if fr_char is not None:
            fr_tok += fr_char
        # else:
        #     assert fr_tok == ""
        
        # if sc_tok_idx is None (deletion), then don't modify cur_tok_idx. We'll stay with the current token.
        if sc_tok_idx is not None:
            cur_tok_idx = sc_tok_idx
    
    assert fr_tok is not None
    if sc_tok_idx is not None:
        if fr_tok == "":
            fr_tok = None
        assert sc_tok_seq[cur_tok_idx] is not None
        assert cur_tok_idx is not None
        fr_toks.append((fr_tok, sc_tok_seq[cur_tok_idx], cur_tok_idx))
    else:
        assert fr_tok != ""
        fr_toks.append((fr_tok, None, None))

    for f, s, i in fr_toks:
        assert (s == None and i == None) or (s != None and i != None)

    return fr_toks

def print_vocab(vocab):
    for fr_tok in vocab.keys():
        print(f"######################### {fr_tok} - TOTAL ({len(vocab[fr_tok])}) #############################")
        for (sc_tok, tok_id), ct in vocab[fr_tok].items():
            print("-------------------------------------------------")
            if sc_tok is None:
                print("NEW FR TOK!!")
            print("ID:", tok_id)
            print("FR_TOK:", fr_tok)
            print("SC_TOK:", sc_tok)
            print("COUNT:", ct)

def make_one2many_selections(fr_vocabulary_with_options):
    print("\nONE2MANY SELECTIONS")
    fr_one2many_selections = {}
    for fr_tok in tqdm(fr_vocabulary_with_options.keys()):
        selected_sc_tok, selected_tok_id, selected_ct = None, None, None
        # Where the same fr_tok occurs with multiple sc_toks, select the most frequent sc_tok. When there are ties, select the sc_tok that matches the fr_tok exactly.
        for (sc_tok, tok_id), ct in fr_vocabulary_with_options[fr_tok].items():
            if (selected_sc_tok, selected_tok_id, selected_ct) == (None, None, None):
                selected_sc_tok = sc_tok
                selected_tok_id = tok_id
                selected_ct = ct
            elif ct > selected_ct:
                selected_sc_tok = sc_tok
                selected_tok_id = tok_id
                selected_ct = ct
            elif ct == selected_ct and sc_tok == fr_tok:
                selected_sc_tok = sc_tok
                selected_tok_id = tok_id
                selected_ct = ct
        
        assert (selected_sc_tok, selected_tok_id, selected_ct) != (None, None, None)
        assert fr_tok not in fr_one2many_selections
        fr_one2many_selections[fr_tok] = (selected_sc_tok, selected_tok_id, selected_ct)
    return fr_one2many_selections

def make_many2one_selections(fr_one2many_selections, next_vocab_id, sc_tokenizer):
    print("MANY2ONE SELECTIONS")
    final_vocabulary = {}

    print("sc_mappings")
    sc_mappings = {}
    for fr_tok, (sc_tok, tok_id, ct) in tqdm(fr_one2many_selections.items()):
        if sc_tok not in sc_mappings:
            sc_mappings[sc_tok] = {}
        assert (fr_tok, tok_id) not in sc_mappings[sc_tok].keys()
        sc_mappings[sc_tok][(fr_tok, tok_id)] = ct
    
    print("selections")
    for sc_tok in tqdm(sc_mappings.keys()):
        STANDARD_tok_id = None
        selected_fr_tok = None
        selected_ct = None
        for (fr_tok, tok_id), ct in sc_mappings[sc_tok].items():
            if STANDARD_tok_id is None:
                STANDARD_tok_id = tok_id
            else:
                assert tok_id == STANDARD_tok_id
            
            if (selected_fr_tok, selected_ct) == (None, None):
                selected_fr_tok = fr_tok
                selected_ct = ct
            elif ct > selected_ct:
                selected_fr_tok = fr_tok
                selected_ct = ct
            elif ct == selected_ct and fr_tok == sc_tok:
                selected_fr_tok = fr_tok
                selected_ct = ct

        assert STANDARD_tok_id is not None
        assert (selected_fr_tok, selected_ct) != (None, None)
        final_vocabulary = add_to_final_vocab(final_vocabulary, selected_fr_tok, STANDARD_tok_id, sc_tokenizer=sc_tokenizer)
        # print("SELECTED FR TOK", selected_fr_tok)

        for (fr_tok, tok_id), ct in sc_mappings[sc_tok].items():
            if fr_tok != selected_fr_tok:
                # print("other FR TOK:", fr_tok)
                final_vocabulary = add_to_final_vocab(final_vocabulary, fr_tok, next_vocab_id, sc_tokenizer=sc_tokenizer)
                next_vocab_id += 1
    return final_vocabulary, next_vocab_id


def add_to_final_vocab(vocab, tok, idx, sc_tokenizer):
    if idx in sc_tokenizer.special_tok_ids:
        assert idx == sc_tokenizer.token2idx[sc_tokenizer.unk]
        print(f"Trying to add `{tok}` as idx `{idx}`, but this is the unk idx. So it's an out of vocab item and will not be added to final vocab.")
        return vocab

    if tok in vocab.keys():
        print("Trying to add", tok, idx)
        print(f"\tBut {tok} set to {vocab[tok]}")
    assert tok not in vocab.keys()

    if idx in vocab.values():
        print(f"Trying to set {tok} to {idx}")
        for v, i in vocab.items():
            if idx == i:
                print(f"\tBut {idx} is already used by another token {v}.")
                break
    assert idx not in vocab.values()
    vocab[tok] = idx
    return vocab


def fr_tokenize(
    sc_tokenizer,
    fr_line,
    sc_line
):
    sc_idx_seq, sc_tok_seq = sc_tokenizer.tokenize(sc_line)
    sc_tok_map = get_tok_map(sc_tok_seq)
    sc_toks_tied = "".join(sc_tok_seq)
    assert len(sc_tok_map) == len(sc_toks_tied) # should just be tuples of each character mapped to token idx then char idx. So it should be the length of the sc_toks_tied ('tied'==concatenated)
    # print(f"SC toks:", sc_tok_seq)
    fr_toks_tied = "▁" + " ".join(fr_line.split()).replace(" ", "▁")

    # fr_toks_tied = "▁Ce▁jour-là,▁sic_la▁bénédiction▁a▁ouvert▁les▁portes▁du▁Ciel▁sur▁Jacob."
    # sc_toks_tied = "▁Se▁zour,▁la▁benediksion▁lrda▁ouver▁les▁por▁di▁Siel▁sir▁Jakob."

    # print("\n")
    # print(f"FR_TOKS_TIED: `{fr_toks_tied}`")
    # print(f"SC_TOKS_TIED: `{sc_toks_tied}`")

    fr2sc_tied_mappings = get_tied_char_alignments(fr_toks_tied, sc_toks_tied)

    fr_toks = get_fr_toks(
        fr2sc_tied_mappings=fr2sc_tied_mappings, 
        fr_toks_tied=fr_toks_tied, 
        sc_tok_map=sc_tok_map,
        sc_tok_seq=sc_tok_seq
    )
    # print("FR TOKS - SC TOKS")
    # for fr_tok, sc_tok, sc_tok_idx in fr_toks:
    #     print(f"{fr_tok}, {sc_tok}, {sc_tok_idx}")

    fr_tok_seq = [
        fr_tok 
        for fr_tok, sc_tok, sc_tok_idx in fr_toks
        if fr_tok is not None
    ]

    assert "".join(fr_tok_seq) == fr_toks_tied
    
    return fr_tok_seq, sc_tok_seq, fr_toks


def get_vocab_distribution_from_tokenizer(tokenizer, data):
    vocab = Counter()
    total_toks = 0
    for line in data:
        idx_seq, toks = tokenizer.tokenize(line)
        for tok in toks:
            if tok not in tokenizer.special_tokens:
                vocab[tok] += 1
                total_toks += 1

    normalized_vocab = {}
    for item, ct in vocab.items():
        assert item not in normalized_vocab
        normalized_vocab[item] = (ct, total_toks)
    return normalized_vocab


def get_vocab_distribution_from_fr_tokenize(tokenizer, fr_data, sc_data):
    assert len(fr_data) == len(sc_data)
    pairs = list(zip(fr_data, sc_data))

    vocab = Counter()
    total_toks = 0
    for fr_line, sc_line in pairs:
        fr_tok_seq, sc_tok_seq, fr_toks = fr_tokenize(
            sc_tokenizer=tokenizer,
            fr_line=fr_line,
            sc_line=sc_line
        )
        for tok in fr_tok_seq:
            if tok not in tokenizer.special_tokens:
                vocab[tok] += 1
                total_toks += 1
    
    normalized_vocab = {}
    for item, ct in vocab.items():
        assert item not in normalized_vocab
        normalized_vocab[item] = (ct, total_toks)
    return normalized_vocab
    

def main(
    fr_file,
    SC_fr2mfe_file,
    mfe_file,
    en_file,
    TOTAL,
    sc_spm_name,
    write_collective_vocab,
    fr_lang,
    mfe_lang,
    en_lang
):
    sc_tokenizer = SPMTokenizer(spm_name=sc_spm_name)
    
    fr_data = read_data(fr_file, TOTAL=TOTAL)
    SC_fr2mfe_data = read_data(SC_fr2mfe_file, TOTAL=TOTAL)
    mfe_data = read_data(mfe_file, TOTAL=TOTAL)
    en_data = read_data(en_file, TOTAL=TOTAL)

    pairs = list(zip(fr_data, SC_fr2mfe_data))

    fr_aligned_vocabulary = {}
    fr_vocab_ids = set()
    next_vocab_id = max(sc_tokenizer.idx2token.keys()) + 1
    assert next_vocab_id == max(sc_tokenizer.token2idx.values()) + 1
    assert next_vocab_id not in sc_tokenizer.idx2token.keys()
    assert next_vocab_id not in sc_tokenizer.token2idx.values()
    print("NEXT VOCAB ID:", next_vocab_id)

    n_diff_num_toks = 0
    n_unk = 0
    n_add_fr = 0

    print("EXAMINING PAIRS")
    for p, (fr_line, sc_line) in tqdm(enumerate(pairs), total=len(pairs)):
        print(f"------------------ ({p}) -------------------")
        print(f"FR_____: `{fr_line}`")
        print(f"SC_____: `{sc_line}`")

        fr_tok_seq, sc_tok_seq, fr_toks = fr_tokenize(
            sc_tokenizer=sc_tokenizer,
            fr_line=fr_line,
            sc_line=sc_line
        )
        print("\n")
        print(f"SC_TOK_SEQ: ({len(sc_tok_seq)})", sc_tok_seq)
        print(f"FR_TOK_SEQ: ({len(fr_tok_seq)})", fr_tok_seq)
        # assert len(fr_tok_seq) == len(sc_tok_seq)
        if len(fr_tok_seq) != len(sc_tok_seq):
            print("\tFR SEQ and SC SEQ have different numbers of tokens!")
            n_diff_num_toks += 1

        # NOW GET FR VOCAB
        for fr_tok, sc_tok, sc_tok_idx in fr_toks:
            if fr_tok not in fr_aligned_vocabulary:
                fr_aligned_vocabulary[fr_tok] = Counter()
            sc_tok_vocab_id = sc_tokenizer.token2idx.get(sc_tok, None) # returns None if sc_tok is None or if sc_tok is an unk token for the tokenizer (out of vocabulary)
            fr_aligned_vocabulary[fr_tok][(sc_tok, sc_tok_vocab_id)] += 1
            fr_vocab_ids.add(sc_tok_vocab_id)
    # For end
    
    print(f"# OCCURENCES WHERE FR AND SC SEQS HAD DIFFERENT NUMBERS OF TOKS: {n_diff_num_toks}")

    print("ALIGNED VOCABULARY")
    print_vocab(fr_aligned_vocabulary)

    print("FR_TOK OPTIONS")
    # fr_vocabulary_with_options -- for each fr_tok, contains the options for sc_tok / new tok
    fr_vocabulary_with_options = {}
    for fr_tok in fr_aligned_vocabulary.keys():
        # print(f"\tFR TOK: `{fr_tok}`")
        assert fr_tok not in fr_vocabulary_with_options
        fr_vocabulary_with_options[fr_tok] = Counter()
        for (sc_tok, tok_id), ct in fr_aligned_vocabulary[fr_tok].items():
            # print(f"\t\tsc_tok: `{sc_tok}`, tok_id: `{tok_id}`")
            if sc_tok is None:
                assert tok_id is None
            if tok_id is None:
                # assert sc_tok is None
                if sc_tok is not None:
                    print("\tTOK_ID is NONE but SC_TOK IS NOT NONE:")
                    print(f"\t\tFR TOK: `{fr_tok}`, sc_tok:`{sc_tok}`, tok_id: `{tok_id}`")
                    tok_id = sc_tokenizer.token2idx[sc_tokenizer.unk]
                    sc_tok = sc_tokenizer.unk
                    # fr_tok = sc_tokenizer.unk
                    print("\t\t-Setting to <unk>:", sc_tokenizer.token2idx['<unk>'], tok_id)
                    n_unk += 1
                else:
                    print("TOK_ID is NONE and SC_TOK is NONE, adding FR TOK")
                    tok_id = next_vocab_id
                    next_vocab_id += 1
                    assert tok_id not in fr_vocab_ids
                    assert tok_id not in sc_tokenizer.idx2token.keys()
                    n_add_fr += 1
                fr_vocab_ids.add(tok_id)
            else:
                assert tok_id in fr_vocab_ids
            
            # if fr_tok == sc_tokenizer.unk:
            #     assert sc_tok == sc_tokenizer.unk
            # if sc_tok == sc_tokenizer.unk:
            #     assert fr_tok == sc_tokenizer.unk

            # if fr_tok != sc_tokenizer.unk:
            #     assert sc_tok != sc_tokenizer.unk
            fr_vocabulary_with_options[fr_tok][(sc_tok, tok_id)] = ct

    print("\n\nUNK TOKS:", n_unk)
    print("ADDED FR TOKS:", n_add_fr)

    print("\n\nVOCABULARY OPTIONS")
    if fr_vocabulary_with_options == fr_aligned_vocabulary:
        print("\tNO NEW FR TOKENS, SO VOCABULARY OPTIONS ARE THE SAME AS FR_ALIGNED_VOCABULARY")
    else:
        print("\tVOCABULARY OPTIONS HAS SOME DIFFERENCES")
        print_vocab(fr_vocabulary_with_options)

    # SELECT ONE FR to MANY SC
    fr_one2many_selections = make_one2many_selections(fr_vocabulary_with_options)

    # SELECT MANY FR to ONE SC. Create new tokens for the remaining FR.
    fr_final_vocabulary, next_vocab_id = make_many2one_selections(fr_one2many_selections, next_vocab_id, sc_tokenizer)
    fr_idx2token = {idx: token for token, idx in fr_final_vocabulary.items()}
    assert len(fr_idx2token) == len(fr_final_vocabulary)

    ordered_fr_vocab = sorted([(idx, token) for token, idx in fr_final_vocabulary.items()])
    ordered_fr_ids = sorted([idx for token, idx in fr_final_vocabulary.items()])
    ordered_sc_vocab = sorted([(idx, token) for token, idx in sc_tokenizer.token2idx.items()])
    ordered_sc_ids = sorted([idx for token, idx in sc_tokenizer.token2idx.items()])


    fr_unique = sorted(list(set(ordered_fr_ids).difference(set(ordered_sc_ids))))
    fr_unique = [(fr_idx2token[idx], idx) for idx in fr_unique]
    sc_unique = sorted(list(set(ordered_sc_ids).difference(set(ordered_fr_ids))))
    sc_unique = [(sc_tokenizer.idx2token[idx], idx) for idx in sc_unique]


    print("------------ FR ------------")
    print("FINAL_FR_VOCAB_SIZE:", len(fr_final_vocabulary), len(ordered_fr_vocab))
    # print_list(ordered_fr_vocab)
    print(f"\nUNIQUE TO FR ({len(fr_unique)})")
    print(f"\nSHARED WITH SC ({len(fr_final_vocabulary) - len(fr_unique)})")
    # print_list(fr_unique)
    print("")
    print("------------ SC ------------")
    print("\nSC_VOCAB_SIZE:", len(sc_tokenizer.token2idx), len(ordered_sc_vocab))
    # print_list(ordered_sc_vocab)
    print(f"\nUNIQUE TO SC ({len(sc_unique)})")
    print(f"\nSHARED WITH FR ({len(sc_tokenizer.token2idx) - len(sc_unique)})")
    # print_list(sc_unique)


    print("\n\n----# Merging Vocabs #----")
    fr_dist = get_vocab_distribution_from_fr_tokenize(
        tokenizer=sc_tokenizer,
        fr_data=fr_data,
        sc_data=SC_fr2mfe_data
    )
    print("FR VOCAB ITEMS:", len(fr_dist))
    mfe_dist = get_vocab_distribution_from_tokenizer(
        tokenizer=sc_tokenizer,
        data=mfe_data
    )
    print("CH VOCAB ITEMS:", len(mfe_dist))
    en_dist = get_vocab_distribution_from_tokenizer(
        tokenizer=sc_tokenizer,
        data=en_data
    )
    print("TG VOCAB ITEMS:", len(en_dist))

    FINAL_COLLECTIVE_VOCAB = {"<LANGS>": {"FR": fr_lang, "CH": mfe_lang, "TG": en_lang}}
    FINAL_COLLECTIVE_VOCAB = add_to_collective_vocab(
        FINAL_COLLECTIVE_VOCAB=FINAL_COLLECTIVE_VOCAB,
        dist=fr_dist,
        token2idx=fr_final_vocabulary,
        div="FR"
    )
    FINAL_COLLECTIVE_VOCAB = add_to_collective_vocab(
        FINAL_COLLECTIVE_VOCAB=FINAL_COLLECTIVE_VOCAB,
        dist=mfe_dist,
        token2idx=sc_tokenizer.token2idx,
        div="CH"
    )
    FINAL_COLLECTIVE_VOCAB = add_to_collective_vocab(
        FINAL_COLLECTIVE_VOCAB=FINAL_COLLECTIVE_VOCAB,
        dist=en_dist,
        token2idx=sc_tokenizer.token2idx,
        div="TG"
    )
    print("FINAL_COLLECTIVE_VOCAB_SIZE:", len(FINAL_COLLECTIVE_VOCAB))

    if os.path.exists(write_collective_vocab):
        assert os.path.isdir(write_collective_vocab)
        print("DELETING", write_collective_vocab)
        shutil.rmtree(write_collective_vocab)
    print("CREATING")
    os.mkdir(write_collective_vocab)
    write_collective_vocab = os.path.join(write_collective_vocab, "FINAL_COLLECTIVE_VOCAB.json")

    print("Writing FINAL_COLLECTIVE_VOCAB to:", write_collective_vocab)
    with open(write_collective_vocab, "w") as outf:
        outf.write(json.dumps(FINAL_COLLECTIVE_VOCAB, ensure_ascii=False, indent=2))

def add_to_collective_vocab(FINAL_COLLECTIVE_VOCAB, dist, token2idx, div="FR"):
    assert div in ["FR", "CH", "TG"] # FR=parent, CH=child, TG=target
    print(f"ADDING {div} DIST TO FINAL_COLLECTIVE_VOCAB")
    div_unknown_items = []
    for item, (ct, total_toks) in tqdm(dist.items()):
        # if item (e.g. the 'ñ' that caused us grief) is not in token2idx, 
        #   it means it probably wasn't added to the fr_final_vocabulary 
        #   (or isn't in the sc_tokenizer vocab, but not sure if that 
        #   will trigger this) and is an unk token. In this case, don't 
        #   add to FINAL_COLLECTIVE_VOCAB either. The dist is built from mere
        #   segmentation, hence we need to check if each tok in the segmenation
        #   is in the vocabulary we decided on earlier.
        if item not in token2idx.keys(): 
            div_unknown_items.append(item)
            continue

        idx = token2idx[item]
        if idx not in FINAL_COLLECTIVE_VOCAB:
            FINAL_COLLECTIVE_VOCAB[idx] = {
                "FR": {"form": None, "ct": 0, "total_toks": 0}, 
                "CH": {"form": None, "ct": 0, "total_toks": 0}, 
                "TG": {"form": None, "ct": 0, "total_toks": 0},
                "TOTAL_CT": 0,
                "TOTAL_TOTAL_TOKS": 0
            }
        assert FINAL_COLLECTIVE_VOCAB[idx][div]["form"] == None
        assert FINAL_COLLECTIVE_VOCAB[idx][div]["ct"] == 0
        assert FINAL_COLLECTIVE_VOCAB[idx][div]["total_toks"] == 0
        FINAL_COLLECTIVE_VOCAB[idx][div]["form"] = item
        FINAL_COLLECTIVE_VOCAB[idx][div]["ct"] = ct
        FINAL_COLLECTIVE_VOCAB[idx][div]["total_toks"] = total_toks
        FINAL_COLLECTIVE_VOCAB[idx]["TOTAL_CT"] += ct
        FINAL_COLLECTIVE_VOCAB[idx]["TOTAL_TOTAL_TOKS"] += total_toks
    print("\tunknown items:", div_unknown_items)
    return FINAL_COLLECTIVE_VOCAB

def print_list(list):
    for item in list:
        print(f"\t- {item}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fr_file", help="parent lang data", default="/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.txt")
    parser.add_argument("--sc_file", help="parent2child lang data", default="/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.SC_FR-MFE-RNN-0_fr2mfe.txt")
    parser.add_argument("--ch_file", help="child lang data")
    parser.add_argument("--tg_file", help="target lang data")
    parser.add_argument("--TOTAL", "-T", type=int, default=None)
    parser.add_argument("--sc_spm_name", default="/home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en")
    parser.add_argument("--write_collective_vocab", "-W", required=True)
    parser.add_argument("--fr_lang", required=True)
    parser.add_argument("--ch_lang", required=True)
    parser.add_argument("--tg_lang", required=True)
    args = parser.parse_args()
    print("- Arguments -")
    for v, k in vars(args).items():
        print(f"\t{v}:{k}")
    return args

if __name__ == "__main__":
    print("###########################")
    print("##    align_tokens.py    ##")
    print("###########################")
    args = get_args()
    main(
        fr_file=args.fr_file,
        SC_fr2mfe_file=args.sc_file,
        mfe_file=args.ch_file,
        en_file=args.tg_file,
        TOTAL=args.TOTAL,
        sc_spm_name=args.sc_spm_name,
        write_collective_vocab=args.write_collective_vocab,
        fr_lang=args.fr_lang,
        mfe_lang=args.ch_lang,
        en_lang=args.tg_lang
    )
