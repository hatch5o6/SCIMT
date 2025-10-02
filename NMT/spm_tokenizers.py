import sentencepiece as spm
import torch
from pytorch_lightning.utilities import rank_zero_info

class SPMTokenizer():
    def __init__(
        self,
        spm_name,
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        VERBOSE=False,
        lang_toks=[]
    ):
        self.spm_name = spm_name
        
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

        self.idx2token = {
            i: self.special_tokens[i]
            for i in range(len(self.special_tokens))
        }
        self.vocab_size = len(self.idx2token)

        if VERBOSE:
            rank_zero_info("SPECIAL TOKS:")
            for key, value in self.idx2token.items():
                rank_zero_info(f"\t- {key}: '{value}'")
            rank_zero_info("---------------")

        resume_tok_count = max(list(self.idx2token.keys())) + 1
        self.spm = spm.SentencePieceProcessor(model_file=spm_name + ".model")
        with open(spm_name + ".vocab") as inf:
            toks = [line.strip().split()[0] for line in inf.readlines()]
        toks = [tok for tok in toks if tok not in self.special_tokens]
        self.idx2token.update({
            resume_tok_count + i: toks[i] 
            for i in range(len(toks))
        })
        self.vocab_size = len(self.idx2token)

        if VERBOSE:
            rank_zero_info("VOCABULARY:")
            for key, value in self.idx2token.items():
                rank_zero_info(f"\t- {key}: '{value}'")
            rank_zero_info("--------------------")

        self.token2idx = {token:i for i, token in self.idx2token.items()}
        assert len(self.idx2token) == len(self.token2idx)
        assert set(self.idx2token.keys()) == set(self.token2idx.values())
        assert set(self.idx2token.values()) == set(self.token2idx.keys())
        assert len(self.idx2token) == len(set(self.idx2token.keys())) == len(set(self.idx2token.values())) == len(set(self.token2idx.keys())) == len(set(self.token2idx.values())) == self.vocab_size
        
        self.special_tok_ids = set([
            self.token2idx[tok]
            for tok in self.special_tokens
        ])

    def tokenize(self, text, lang=None, add_lang_token=False, return_tensor=False, add_special=True):
        if add_lang_token is True:
            assert lang is not None
        if lang is not None:
            assert f"<{lang}>" in self.lang_toks_set
        spm_sequence = self.spm.encode(text.strip(), out_type=str)
        if lang is not None and add_lang_token == True:
            spm_sequence = [f"<{lang}>"] + spm_sequence
        idx_sequence = []
        # TODO Do I need to have it add a bos token? -- Nope! According to ChatGPT anyway :)
        for token in spm_sequence:
            if token not in self.token2idx:
                token = self.unk
            idx_sequence.append(self.token2idx[token])
        if add_special:
            spm_sequence.append(self.eos) # add eos token
            idx_sequence.append(self.token2idx[self.eos]) # add eos token
        if return_tensor:
            idx_sequence = torch.tensor(idx_sequence)
        return idx_sequence, spm_sequence


    def batch_tokenize(self, batch, pad_batch=True, return_tensor=False, add_lang_token=False, add_special=True, batch_has_langs=False):
        # tokenize
        if batch_has_langs:
            batch = [
                self.tokenize(line, lang=lang, add_lang_token=add_lang_token, add_special=add_special)[0]
                for line, lang in batch
            ]
        else:
            batch = [
                self.tokenize(line, lang=None, add_lang_token=add_lang_token, add_special=add_special)[0]
                for line in batch
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

    def detokenize(self, token_ids, remove_special_toks=False):
        if remove_special_toks:
            token_ids = [
                tok_id 
                for tok_id in token_ids
                if tok_id not in self.special_tok_ids
            ]

        decoded_toks = []
        for tok_id in token_ids:
            if tok_id in self.idx2token:
                tok = self.idx2token[tok_id]
            else:
                tok = self.unk
            decoded_toks.append(tok)
        decoded_line = "".join(decoded_toks).replace("▁", " ")

        return decoded_line

    def batch_detokenize(self, batch, remove_special_toks=False):
        batch = [
            self.detokenize(seq, remove_special_toks=remove_special_toks)
            for seq in batch
        ]
        return batch
