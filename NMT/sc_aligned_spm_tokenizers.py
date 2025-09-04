import sentencepiece as spm
import torch
import json
from align_tokens import fr_tokenize
from spm_tokenizers import SPMTokenizer
import math
from tqdm import tqdm

class SCAlignedSPMTokenizer():
    def __init__(
        self,
        pl,
        cl,
        tl,
        # fr_spm_name,
        sc_spm_name,
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        VERBOSE=False,
        aligned_vocab="", # json file, created by align_tokens.py,
        lang_toks=[],
        VOCAB_SIZE_CAP=math.inf
    ):
        self.pl = pl
        self.cl = cl
        self.tl = tl
        # self.fr_spm_name = fr_spm_name

        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.lang_toks = []
        for tok in lang_toks:
            if tok not in self.lang_toks:
                self.lang_toks.append(tok)
        self.lang_toks_set = set(self.lang_toks)
        self.special_tokens = [
            self.bos,
            self.eos,
            self.unk,
            self.pad
        ] + lang_toks

        self.idx2token = {}
        for t, tok in enumerate(self.special_tokens):
            for l in self.lang_toks:
                self.add_to_idx2token(
                    idx=t,
                    lang=l,
                    tok=tok
                )

        if VERBOSE:
            print("ALIGNED SPECIAL TOKS:")
            for key, value in self.idx2token.items():
                print(f"\t- {key}: '{value}'")
            print("---------------")

        self.sc_tokenizer = SPMTokenizer(
            spm_name=sc_spm_name,
            bos=self.bos,
            eos=self.eos,
            unk=self.unk,
            pad=self.pad,
            VERBOSE=False
        )

        print("READING IN ALIGNED VOCAB")
        print("\tVOCAB_SIZE_CAP:", VOCAB_SIZE_CAP)
        TOKENS_IN_EACH_LANG = self.read_aligned_vocab(
            aligned_vocab_f=aligned_vocab, 
            VOCAB_SIZE_CAP=VOCAB_SIZE_CAP
        )
        print("VOCAB_SIZE:", len(self.idx2token))
        assert len(self.idx2token) <= VOCAB_SIZE_CAP
        self.vocab_size = len(self.idx2token)

        print("\n\nTOKENS IN EACH LANG:")
        for lang_label, ct in TOKENS_IN_EACH_LANG.items():
            print(f"\t{lang_label}: {ct}, {ct / len(self.idx2token)}")

        if VERBOSE:
            print(f"ALIGNED VOCABULARY ({self.vocab_size}):")
            for key, value in self.idx2token.items():
                print(f"\t- {key}: '{value}'")
            print("--------------------")
        
        # self.token2idx = {token:i for i, token in self.idx2token.items()}
        # assert len(self.idx2token) == len(self.token2idx)
        # assert set(self.idx2token.keys()) == set(self.token2idx.values())
        # assert set(self.idx2token.values()) == set(self.token2idx.keys())
        # assert len(self.idx2token) == len(set(self.idx2token.keys())) == len(set(self.idx2token.values())) == len(set(self.token2idx.keys())) == len(set(self.token2idx.values())) == self.vocab_size
        
        self.token2idx = self.make_token2idx()
        self.special_tok_ids = set([
            self.token2idx[tok][self.lang_toks[0]]
            for tok in self.special_tokens
        ])

    def add_to_idx2token(self, idx, lang, tok):
        # print("adding", idx, lang, tok)
        assert isinstance(idx, int)
        assert isinstance(lang, str)
        assert isinstance(tok, str)
        assert idx is not None
        assert lang is not None
        assert tok is not None

        if idx not in self.idx2token:
            self.idx2token[idx] = {l: None for l in self.lang_toks}
        if self.idx2token[idx][lang] != None:
            print("Trying to add", idx, lang, tok)
            print("\tTrying to set self.idx2token[{idx}][{lang}]")
            print(f"\tself.idx2token[{idx}][{lang}] already equals {self.idx2token[idx][lang]}")
        assert self.idx2token[idx][lang] == None
        self.idx2token[idx][lang] = tok

    def make_token2idx(self):
        with open("sc_vocab_idx2token.json", "w") as outf:
            outf.write(json.dumps(self.idx2token, ensure_ascii=False, indent=2))
        token2idx = {}
        for idx, langs in self.idx2token.items():
            for lang, form in langs.items():
                if form != None:
                    if form not in token2idx:
                        token2idx[form] = {l: None for l in self.lang_toks}
                    if token2idx[form][lang] != None:
                        print("Trying to set form `{form}`, lang `{lang}` to {idx}, but it's already set to {token2idx[form][lang}")
                    assert token2idx[form][lang] == None
                    token2idx[form][lang] = idx
        assert None not in token2idx.keys()
        with open("sc_vocab_token2idx.json", "w") as outf:
            outf.write(json.dumps(token2idx, ensure_ascii=False, indent=2))
        return token2idx

    def read_aligned_vocab(self, aligned_vocab_f, VOCAB_SIZE_CAP):
        with open(aligned_vocab_f) as inf:
            aligned_vocab = json.load(inf)
        aligned_langs = aligned_vocab["<LANGS>"]
        tokenizer_langs = [ltok[1:-1] for ltok in self.lang_toks]
        
        if not (sorted(list(aligned_langs.values())) == sorted(tokenizer_langs)):
            print("ERROR: Aligned vocab langs do not match tokenizer langs")
            print("ALIGNED_VOCAB LANGS:", sorted(list(aligned_langs.values())))
            print("TOKENIZER LANGS (TOKS):", sorted(self.lang_toks))
            assert False
        
        # Get all idxs in a list, get their norm_dist
        # Then sort the list by norm dist 
        # and truncate it at VOCAB_SIZE_CAP 
        print("Getting accepted idxs")
        idxs = []
        for idx, divs in aligned_vocab.items():
            if idx == "<LANGS>": continue
            assert isinstance(idx, str)
            idx = int(idx)
            total_ct = divs["TOTAL_CT"]
            total_total_toks = divs["TOTAL_TOTAL_TOKS"]
            norm_dist = total_ct / total_total_toks
            idxs.append((norm_dist, idx))
        idxs.sort(reverse=True)
        if VOCAB_SIZE_CAP < math.inf:
            # need to adjust the cap to account for the special tokens already in the vocab
            accepted_idxs = idxs[:VOCAB_SIZE_CAP - len(self.special_tokens)]
        else:
            accepted_idxs = idxs
        accepted_idxs = [idx for norm_dist, idx in accepted_idxs]

        print("Getting vocab")
        TOKENS_IN_EACH_LANG = {}
        for idx, divs in tqdm(aligned_vocab.items()):
            if idx == "<LANGS>": continue
            assert isinstance(idx, str)
            idx = int(idx)
            if idx in accepted_idxs:
                for div, div_data in divs.items():
                    if div in ["TOTAL_CT", "TOTAL_TOTAL_TOKS"]: continue
                    div_lang = aligned_langs[div]
                    form = div_data["form"]
                    if form is not None:
                        self.add_to_idx2token(idx=idx, lang=f"<{div_lang}>", tok=form)

                    # log number of tokens for each lang
                    lang_label = f"{div}-{div_lang}"
                    if lang_label not in TOKENS_IN_EACH_LANG:
                        TOKENS_IN_EACH_LANG[lang_label] = 0
                    TOKENS_IN_EACH_LANG[lang_label] += 1
        return TOKENS_IN_EACH_LANG
                
    def read_json(f):
        with open(f) as inf:
            data = json.load(inf)
        return data

    def tokenize(self, fr_text, sc_text, lang, return_tensor=False, add_lang_token=False, add_special=False):
        lang_tok = f"<{lang}>"
        assert lang_tok in self.lang_toks_set
        assert lang_tok in self.token2idx
        if sc_text is not None:
            assert lang_tok == self.pl
            fr_tok_seq, _, _ = fr_tokenize(
                sc_tokenizer=self.sc_tokenizer,
                fr_line=fr_text,
                sc_line=sc_text
            )
        else:
            assert lang_tok in [self.cl, self.tl]
            fr_tok_seq = self.sc_tokenizer.tokenize(fr_text)[1]

        if add_lang_token == True:
            fr_tok_seq = [lang_tok] + fr_tok_seq
        
        for f, token in enumerate(fr_tok_seq):
            if token in self.token2idx:
                # assert that the token has an id for the lang
                if self.token2idx[token][lang_tok] == None:
                    # print(f"ALERT: form for lang {lang_tok} has None idx")
                    # print(fr_tok_seq)
                    # print(f"Has None idx: ({f}) `{token}`")
                    # print(self.token2idx[token])
                    # print(f"Will set token to `{self.unk}`")
                    token = self.unk
                    fr_tok_seq[f] = token
                assert self.token2idx[token][lang_tok] != None
            elif token not in self.token2idx:
                # if token is not in the vocab, then we make it an unk token
                fr_tok_seq[f] = self.unk
        
        fr_idx_sequence = []
        for token in fr_tok_seq:
            tokidx = self.token2idx[token][lang_tok]
            assert tokidx is not None
            fr_idx_sequence.append(tokidx)
        if add_special:
            fr_tok_seq.append(self.eos)
            fr_idx_sequence.append(self.token2idx[self.eos][lang_tok]) # add eos token
        if return_tensor:
            fr_idx_sequence = torch.tensor(fr_idx_sequence)
        return fr_idx_sequence, fr_tok_seq

    def batch_tokenize(self, batch, pad_batch=True, return_tensor=False, add_lang_token=True, add_special=False):
        # tokenize
        batch = [
            self.tokenize(fr_text=fr_line, sc_text=sc_line, lang=lang, add_lang_token=add_lang_token, add_special=add_special)[0]
            for fr_line, sc_line, lang in batch
        ]
        # pad sequences
        if pad_batch:
            pad_tok_idx = self.token2idx[self.pad][self.lang_toks[0]]
            for lx in self.lang_toks:
                assert self.token2idx[self.pad][lx] == pad_tok_idx
            max_seq = max([len(seq) for seq in batch])
            for s, seq in enumerate(batch):
                while len(seq) < max_seq:
                    seq.append(pad_tok_idx)
                batch[s] = seq
        # make tensor
        if return_tensor:
            batch = torch.tensor(batch)
        return batch

    def detokenize(self, token_ids, lang, remove_special_toks=False):
        if remove_special_toks:
            token_ids = [
                tok_id 
                for tok_id in token_ids
                if tok_id not in self.special_tok_ids
            ]

        decoded_toks = []
        for tok_id in token_ids:
            if tok_id in self.idx2token:
                tok = self.idx2token[tok_id][lang]
                if tok is None:
                    # if model generates an id that isn't a token in the tgt language -- this is possible, so we should anticipate it.
                    #TODO maybe create criteria to choose the surface form from one of the other languages. How though?
                    # tok = self.unk
                    tok = f"<TOK_ID_{tok_id}>"
            else:
                assert False # this shouldn't ever happen. Essentially, this expects the model to generate an out of vocab id.
                tok = self.unk
            decoded_toks.append(tok)
        decoded_line = "".join(decoded_toks).replace("▁", " ")

        return decoded_line

    def batch_detokenize(self, batch, remove_special_toks=False):
        batch = [
            self.detokenize(seq, lang=lang, remove_special_toks=remove_special_toks)
            for seq, lang in batch
        ]
        return batch


"""
BATCH --------
[38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857] len: 13
[427, 3178, 40520, 7905, 32685, 7896, 73768, 31857] len: 8
[910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857] len: 23
NORMAL -------
[38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857]
['▁Des', '▁sites', ',', '▁seules', '▁les', '▁vieilles', '▁batailles', '▁et', '▁ils', '▁lui', '▁ont', '▁dit', '.']
-
[427, 3178, 40520, 7905, 32685, 7896, 73768, 31857]
['▁Le', '▁vrai', '▁musulman', '▁croit', '▁aux', '▁anges', "▁d'Allah", '.']
-
[910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857]
['▁Et', '▁el', 'le▁n', "'", 'a', '▁pas', '▁expliqué', '▁non', '▁plus', '▁pourquoi', '▁votre', '▁smartphone', '▁est', '▁tombé', '▁plus', '▁de', '▁fois', '▁que', '▁le', '▁reste', '▁de', "▁l'humanité", '.']
-


ADD_LANG_TOK
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857] len: 14
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857] len: 9
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857] len: 24
NORMAL -------
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857]
['<fr>', '▁Des', '▁sites', ',', '▁seules', '▁les', '▁vieilles', '▁batailles', '▁et', '▁ils', '▁lui', '▁ont', '▁dit', '.']
-
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857]
['<fr>', '▁Le', '▁vrai', '▁musulman', '▁croit', '▁aux', '▁anges', "▁d'Allah", '.']
-
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857]
['<fr>', '▁Et', '▁el', 'le▁n', "'", 'a', '▁pas', '▁expliqué', '▁non', '▁plus', '▁pourquoi', '▁votre', '▁smartphone', '▁est', '▁tombé', '▁plus', '▁de', '▁fois', '▁que', '▁le', '▁reste', '▁de', "▁l'humanité", '.']


ADD_SPECIAL
BATCH --------
[38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1] len: 14
[427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1] len: 9
[910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1] len: 24
NORMAL -------
[38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1]
['▁Des', '▁sites', ',', '▁seules', '▁les', '▁vieilles', '▁batailles', '▁et', '▁ils', '▁lui', '▁ont', '▁dit', '.', '</s>']
-
[427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1]
['▁Le', '▁vrai', '▁musulman', '▁croit', '▁aux', '▁anges', "▁d'Allah", '.', '</s>']
-
[910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1]
['▁Et', '▁el', 'le▁n', "'", 'a', '▁pas', '▁expliqué', '▁non', '▁plus', '▁pourquoi', '▁votre', '▁smartphone', '▁est', '▁tombé', '▁plus', '▁de', '▁fois', '▁que', '▁le', '▁reste', '▁de', "▁l'humanité", '.', '</s>']



ADD_LANG_TOK + ADD_SPECIAL
BATCH --------
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1] len: 15
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1] len: 10
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1] len: 25
NORMAL -------
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1]
['<fr>', '▁Des', '▁sites', ',', '▁seules', '▁les', '▁vieilles', '▁batailles', '▁et', '▁ils', '▁lui', '▁ont', '▁dit', '.', '</s>']
-
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1]
['<fr>', '▁Le', '▁vrai', '▁musulman', '▁croit', '▁aux', '▁anges', "▁d'Allah", '.', '</s>']
-
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1]
['<fr>', '▁Et', '▁el', 'le▁n', "'", 'a', '▁pas', '▁expliqué', '▁non', '▁plus', '▁pourquoi', '▁votre', '▁smartphone', '▁est', '▁tombé', '▁plus', '▁de', '▁fois', '▁que', '▁le', '▁reste', '▁de', "▁l'humanité", '.', '</s>']


ADD_LANG_TOK + ADD_SPECIAL + PAD
BATCH --------
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] len: 25
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] len: 25
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1] len: 25
NORMAL -------
[4, 38246, 39598, 31862, 43504, 106, 55606, 34303, 93, 32775, 36449, 90, 988, 31857, 1]
['<fr>', '▁Des', '▁sites', ',', '▁seules', '▁les', '▁vieilles', '▁batailles', '▁et', '▁ils', '▁lui', '▁ont', '▁dit', '.', '</s>']
-
[4, 427, 3178, 40520, 7905, 32685, 7896, 73768, 31857, 1]
['<fr>', '▁Le', '▁vrai', '▁musulman', '▁croit', '▁aux', '▁anges', "▁d'Allah", '.', '</s>']
-
[4, 910, 38296, 32787, 31869, 365, 165, 47353, 457, 262, 1890, 562, 19226, 134, 42065, 262, 29, 1128, 100, 117, 235, 29, 6579, 31857, 1]
['<fr>', '▁Et', '▁el', 'le▁n', "'", 'a', '▁pas', '▁expliqué', '▁non', '▁plus', '▁pourquoi', '▁votre', '▁smartphone', '▁est', '▁tombé', '▁plus', '▁de', '▁fois', '▁que', '▁le', '▁reste', '▁de', "▁l'humanité", '.', '</s>']



CL TOKENS:
OG BATCH------
[5, 2711, 1072, 783, 31854, 1, 3]
[5, 4934, 31876, 31839, 1506, 31854, 1]
[5, 5348, 28332, 413, 31854, 1, 3]

SC BATCH------
[5, 12913, 322, 31841, 31857, 1, 3]
[5, 14356, 31882, 31843, 1472, 31857, 1]
[5, 5446, 19154, 88, 31857, 1, 3]

BATCH --------
[5, 12913, 322, 31841, 31857, 1, 3] len: 7
[5, 14356, 31882, 31843, 1472, 31857, 1] len: 7
[5, 5446, 19154, 88, 31857, 1, 3] len: 7



"""