import os
import csv
import torch
from tqdm import tqdm

class CharacterTokenizer():
    def __init__(
        self,
        vocab_path="char_cache/vocab.en.csv",
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
        data_paths=[],
        lang_toks=[],
        OFFSET=0
    ):
        vocab_dir = "/".join(vocab_path.split("/")[:-1])
        if not os.path.exists(vocab_dir):
            os.mkdir(vocab_dir)
        
        self.vocab_path = vocab_path

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
        if OFFSET > 0:
            i = OFFSET
        else:
            i = max(self.idx2token.keys()) + 1
        
        if not os.path.exists(vocab_path):
            assert data_paths != []
            progress = tqdm(total=10000000)
            for data_path in data_paths:
                print("READING CHARS FROM", data_path)
                with open(data_path) as inf:
                    line = inf.readline()
                    while line:
                        progress.update(1)
                        for char in line:
                            if char not in self.idx2token.values():
                                self.idx2token[i] = char
                                i += 1
                        line = inf.readline()
            self.token2idx = {
                token: i
                for i, token in self.idx2token.items()
            }
            self.write_new_tokens()
        else:
            print("READING FROM VOCAB", vocab_path)
            with open(vocab_path, newline='') as inf:
                reader = csv.reader(inf)
                for row in reader:
                    token, idx = tuple(row)
                    idx = int(idx)
                    assert idx not in self.idx2token or self.idx2token.get(idx) == token
                    self.idx2token[idx] = token
            self.token2idx = {
                token: i
                for i, token in self.idx2token.items()
            }

        print("idx2token")
        print(self.idx2token)

        print("token2idx")
        print(self.token2idx)
        self.next_tok_id = max(list(self.idx2token.keys())) + 1
        self.vocab_size = len(self.token2idx)
        print("VOCAB SIZE", self.vocab_size)

        self.special_tok_ids = set([self.token2idx[tok] for tok in self.special_tokens])
    
    def tokenize(self, text, lang=None, add_lang_token=False, return_tensor=False, add_special=True):
        if add_lang_token is True:
            assert lang is not None
        if lang is not None:
            assert f"<{lang}>" in self.lang_toks_set

        tokens = list(text)
        if lang is not None and add_lang_token == True:
            tokens = [f"<{lang}>"] + tokens
        for c, char in enumerate(tokens):
            if char not in self.token2idx:
                tokens[c] = self.unk
        if add_special:
            tokens.append(self.eos)
        token_ids = [self.token2idx[char] for char in tokens]
        if return_tensor:
            token_ids = torch.tensor(token_ids)

        return token_ids, tokens
    
    def batch_tokenize(self, batch, pad_batch=True, return_tensor=False, add_lang_token=False, add_special=True, batch_has_langs=False):
        # tokenize
        batch = list(batch)
        if batch_has_langs:
            tokenized_batch = [
                self.tokenize(line, lang=lang, add_lang_token=add_lang_token, add_special=add_special)[0]
                for line, lang in batch
            ]
        else:
            tokenized_batch = [
                self.tokenize(line, lang=None, add_lang_token=add_lang_token, add_special=add_special)[0]
                for line in batch
            ]

        # pad sequences
        if pad_batch:
            max_seq = max([len(seq) for seq in tokenized_batch])
            for s, seq in enumerate(tokenized_batch):
                while len(seq) < max_seq:
                    seq.append(self.token2idx[self.pad])
                tokenized_batch[s] = seq

        if return_tensor:
            tokenized_batch = torch.tensor(tokenized_batch)
        return tokenized_batch

    def detokenize(self, sequence, remove_special_toks=False):
        if remove_special_toks:
            sequence = [
                tok_id
                for tok_id in sequence
                if tok_id not in self.special_tok_ids
            ]
        for s, idx in enumerate(sequence):
            if idx not in self.idx2token:
                sequence[s] = self.token2idx[self.unk]
        tokens = [
            self.idx2token[idx]
            for idx in sequence
        ]
        return "".join(tokens)
    
    def batch_detokenize(self, batch, remove_special_toks=False):
        batch = [
            self.detokenize(seq, remove_special_toks=remove_special_toks)
            for seq in batch
        ]
        return batch

    def write_new_tokens(self):
        print("WRITING VOCAB TO", self.vocab_path)
        with open(self.vocab_path, "w", newline='') as outf:
            for token, idx in self.token2idx.items():
                writer = csv.writer(outf)
                writer.writerow([token, idx])