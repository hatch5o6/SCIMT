import sentencepiece as spm
import torch
import json
from align_tokens import fr_tokenize
from spm_tokenizers import SPMTokenizer
import math

class SCAlignedSPMTokenizer():
    def __init__(
        self,
        fr_spm_name,
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
        self.fr_spm_name = fr_spm_name

        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.lang_toks = []
        for tok in lang_toks:
            if tok not in self.lang_toks:
                self.lang_toks.append(tok)
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
        self.read_aligned_vocab(
            aligned_vocab=aligned_vocab, 
            VOCAB_SIZE_CAP=VOCAB_SIZE_CAP
        )
        print("VOCAB_SIZE:", len(self.idx2token))
        assert len(self.idx2token) <= VOCAB_SIZE_CAP
        self.vocab_size = len(self.idx2token)

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
        if idx not in self.idx2token:
            self.idx2token[idx] = {l: None for l in self.lang_toks}
        assert self.idx2token[idx][lang] == None
        self.idx2token[idx][lang] = tok

    def make_token2idx(self):
        token2idx = {}
        for idx, langs in self.idx2token:
            for lang, form in langs.items():
                if form not in token2idx:
                    token2idx[form] = {l: None for l in self.lang_toks}
                assert token2idx[form][lang] == None
                token2idx[form][lang] = idx
        return token2idx

    def read_aligned_vocab(self, aligned_vocab_f, VOCAB_SIZE_CAP):
        with open(aligned_vocab_f) as inf:
            aligned_vocab = json.load(inf)
        aligned_langs = aligned_vocab["<LANGS>"]
        assert sorted(list(aligned_langs.values())) == sorted(self.lang_toks)
        
        # Get all idxs in a list, get their norm_dist
        # Then sort the list by norm dist 
        # and truncate it at VOCAB_SIZE_CAP 
        idxs = []
        for idx, divs in aligned_vocab.items():
            if idx == "<LANGS>": continue

            total_ct = divs["TOTAL_CT"]
            total_total_toks = divs["TOTAL_TOTAL_TOKS"]
            norm_dist = total_ct / total_total_toks
            idxs.append((norm_dist, idx))
        idxs.sort(reverse=True)
        accepted_idxs = idxs[:VOCAB_SIZE_CAP]

        for idx, divs in aligned_vocab.items():
            if idx == "<LANGS>": continue
            if idx in accepted_idxs:
                for div, div_data in divs.items():
                    if div in ["TOTAL_CT", "TOTAL_TOTAL_TOKS"]: continue
                    div_lang = aligned_langs[div]
                    form = div_data["form"]
                    self.add_to_idx2token(idx=idx, lang=div_lang, tok=form)
                
    def read_json(f):
        with open(f) as inf:
            data = json.load(inf)
        return data

    def sc_align_tokenize(self, fr_text, sc_text, lang, return_tensor=False, add_special=False):
        if sc_text is not None:
            fr_tok_seq, _, _ = fr_tokenize(
                sc_tokenizer=self.sc_tokenizer,
                fr_line=fr_text,
                sc_line=sc_text
            )
        else:
            fr_tok_seq = 
        fr_idx_sequence = []
        for token in fr_tok_seq:
            if token not in self.token2idx:
                token = self.unk
            fr_idx_sequence.append(self.token2idx[token][lang])
        if add_special:
            fr_tok_seq.append(self.eos)
            fr_idx_sequence.append(self.token2idx[self.eos][lang]) # add eos token
        if return_tensor:
            fr_idx_sequence = torch.tensor(fr_idx_sequence)
        return fr_idx_sequence, fr_tok_seq

    def sc_align_batch_tokenize(self, batch, pad_batch=True, return_tensor=False, add_special=False):
        # tokenize
        batch = [
            self.sc_align_tokenize(fr_text=fr_line, sc_text=sc_line, lang=lang, add_special=add_special)[0]
            for fr_line, sc_line, lang in batch
        ]
        # pad sequences
        if pad_batch:
            max_seq = max([len(seq) for seq in batch])
            for s, seq in enumerate(batch):
                while len(seq) < max_seq:
                    seq.append(self.token2idx[self.pad])
                batch[s] = seq
        # make tensor
        if return_tensor:
            batch = torch.tensor(batch)
        return batch

    def sc_align_detokenize(self, token_ids, lang, remove_special_toks=False):
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
                    tok = self.unk
            else:
                tok = self.unk
            decoded_toks.append(tok)
        decoded_line = "".join(decoded_toks).replace("▁", " ")

        return decoded_line

    def sc_align_batch_detokenize(self, batch, remove_special_toks=False):
        batch = [
            self.sc_align_detokenize(seq, lang=lang, remove_special_toks=remove_special_toks)
            for seq, lang in batch
        ]
        return batch

    