from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import os
import json
import argparse
import yaml

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from params import *

# Used code from https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class Lang:
    def __init__(self, name, dir_path):
        self.name = name
        self.token2index = {}
        self.token2count = {}
        self.index2token = {0: "SOS", 1: "EOS"}
        self.n_tokens = 2  # Count SOS and EOS
        self.dir_path = dir_path
        print("DIR PATH:", self.dir_path)
        self.idx2token_path = os.path.join(self.dir_path, "idx2token.json")
        self.token2count_path = os.path.join(self.dir_path, "token2count.json")

        if not os.path.exists(self.dir_path):
            print("Creating", self.dir_path)
            os.mkdir(self.dir_path)
        else:
            print("Reading", self.idx2token_path)
            with open(self.idx2token_path) as inf:
                index2token_dict = json.load(inf)
                for k, v in index2token_dict.items():
                    int_k = int(k)
                    if int_k in self.index2token:
                        assert int_k in [0, 1]
                        assert self.index2token[int_k] == v
                    else:
                        self.index2token[int_k] = v
                for idx, tok in self.index2token.items():
                    assert tok not in self.token2index
                    self.token2index[tok] = idx
                self.n_tokens = len(self.index2token)
                for idx in self.index2token:
                    assert idx < self.n_tokens
            print("Reading", self.token2count_path)
            with open(self.token2count_path) as inf:
                self.token2count = json.load(inf)

    def addSequence(self, sequence):
        for token in sequence:
            self.addToken(token)

    def addToken(self, token):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1
    
    def save(self):
        with open(self.idx2token_path, "w") as outf:
            outf.write(json.dumps(self.index2token, ensure_ascii=False, indent=2))
        with open(self.token2count_path, "w") as outf:
            outf.write(json.dumps(self.token2count, ensure_ascii=False, indent=2))
        return self.dir_path


def readData(file_path, reverse=False, LIMIT="None"):
    assert LIMIT == "None" or isinstance(LIMIT, int)
    pairs = []
    with open(file_path) as inf:
        for line in inf:
            d1, d2, _ = line.strip().split(" ||| ")
            pairs.append((d1.strip(), d2.strip()))
            if LIMIT != "None" and len(pairs) >= LIMIT:
                print(f"Ending LIMIT={LIMIT}")
                break
    
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    pairs = [tuple(p) for p in pairs]
    print("PAIRS")
    for pair in pairs[:5]:
        print(pair)

    return pairs


def prepareData(file_path, lang1, lang2, input_lang_dir, output_lang_dir, reverse=False, LIMIT="None"):
    assert LIMIT == "None" or isinstance(LIMIT, int)
    pairs = readData(file_path, reverse=reverse, LIMIT=LIMIT)

    input_lang = Lang(lang1, dir_path=input_lang_dir)
    output_lang = Lang(lang2, dir_path=output_lang_dir)
    print(f"Read {len(pairs)} sequence pairs")
    for src_seq, tgt_seq in pairs:
        input_lang.addSequence(src_seq)
        output_lang.addSequence(tgt_seq)
    print("Total tokens:")
    print(f"\t{input_lang.name}: {input_lang.n_tokens}")
    print(f"\t{output_lang.name}: {output_lang.n_tokens}")
    
    return input_lang, output_lang, pairs


def indexesFromSequence(lang, sequence):
    return [lang.token2index[c] for c in sequence]

def tensorFromSequence(lang, sequence, device):
    indexes = indexesFromSequence(lang, sequence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSequence(input_lang, pair[0])
    target_tensor = tensorFromSequence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_unique_chars(pairs):
    unique_chars_1 = set()
    unique_chars_2 = set()
    for seq1, seq2 in pairs:
        for char in seq1:
            unique_chars_1.add(char)
        for char in seq2:
            unique_chars_2.add(char)
    return unique_chars_1, unique_chars_2

def check_unique_chars_in_lang(unique_chars, lang):
    print("Checking unique chars in input are in input_lang", lang.name)
    for char in unique_chars:
        if char not in lang.token2index.keys():
            raise ValueError(f"Character '{char}' not in Lang {lang.name}!")

def get_dataloader(file_path, lang1, lang2, input_lang_dir, output_lang_dir, reverse=True, shuffle=True, batch_size=32, device="cuda", LIMIT="None"):
    assert LIMIT == "None" or isinstance(LIMIT, int)
    pairs = readData(file_path, reverse=reverse, LIMIT=LIMIT)

    input_lang = Lang(name=lang1, dir_path=input_lang_dir)
    output_lang = Lang(name=lang2, dir_path=output_lang_dir)

    # Check that unique chars in file_path are in corresponding lang
    input_unique_chars, output_unique_chars = get_unique_chars(pairs)
    check_unique_chars_in_lang(input_unique_chars, input_lang)
    check_unique_chars_in_lang(output_unique_chars, output_lang)

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSequence(input_lang, inp)
        tgt_ids = indexesFromSequence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    # train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, 
        # sampler=train_sampler,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return input_lang, output_lang, train_dataloader

def create_langs(config):
    data = ""
    for data_file in ["train", "val", "test"]:
        with open(config[data_file]) as inf:
            data += inf.read().strip() + "\n"
    with open(config["lang_data"], "w") as outf:
        outf.write(data)

    input_lang, output_lang, pairs = prepareData(
        file_path=config["lang_data"], 
        lang1=config["src"],
        lang2=config["tgt"], 
        input_lang_dir=config["src_lang"], 
        output_lang_dir=config["tgt_lang"],
        reverse=True, 
        LIMIT="None"
    )

    input_lang.save()
    output_lang.save()

# def make_lang_dir(config, lang_type):
#     assert lang_type in ["src", "tgt"]
#     lang_dir = config[f"{lang_type}_lang"]
#     if os.path.exists(lang_dir):
#         raise ValueError(f"lang_dir {lang_dir} already exists!")
#     os.mkdir(lang_dir)

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = read_config(args.config)
    create_langs(config)