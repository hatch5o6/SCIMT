import sentencepiece as spm
import torch

class SPMTokenizer():
    def __init__(
        self,
        spm_name,
        bos="<s>",
        eos="</s>",
        unk="<unk>",
        pad="<pad>",
    ):
        self.bos = bos
        self.eos = eos
        self.unk = unk
        self.pad = pad
        self.special_tokens = [
            self.bos,
            self.eos,
            self.unk,
            self.pad
        ]

        self.idx2token = {
            i: self.special_tokens[i]
            for i in range(len(self.special_tokens))
        }
        print("SPECIAL TOKS:")
        self.vocab_size = len(self.idx2token)
        for key, value in self.idx2token.items():
            print(f"\t- {key}: '{value}'")
        print("---------------")

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

        print("VOCABULARY:")
        for key, value in self.idx2token.items():
            print(f"\t- {key}: '{value}'")
        print("--------------------")

        self.token2idx = {token:i for i, token in self.idx2token.items()}
        assert len(self.idx2token) == len(self.token2idx)
        assert set(self.idx2token.keys()) == set(self.token2idx.values())
        assert set(self.idx2token.values()) == set(self.token2idx.keys())
        self.special_tok_ids = set([
            self.token2idx[tok]
            for tok in self.special_tokens
        ])

    def tokenize(self, text, return_tensor=False):
        spm_sequence = self.spm.encode(text.strip(), out_type=str)
        idx_sequence = []
        # TODO Do I need to have it add a bos token?
        for token in spm_sequence:
            if token not in self.token2idx:
                token = self.unk
            idx_sequence.append(self.token2idx[token])
        idx_sequence.append(self.token2idx[self.eos]) # add eos token
        if return_tensor:
            idx_sequence = torch.tensor(idx_sequence)
        return idx_sequence, spm_sequence


    def batch_tokenize(self, batch, return_tensor=False):
        # tokenize
        batch = [
            self.tokenize(line)[0]
            for line in batch
        ]
        # pad sequences
        max_seq = max([len(seq) for seq in batch])
        for s, seq in enumerate(batch):
            while len(seq) < max_seq:
                seq.append(self.token2idx[self.pad])
            batch[s] = seq
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
                tok = str(tok_id)
            decoded_toks.append(tok)
        decoded_line = "".join(decoded_toks).replace("▁", " ")

        return decoded_line

    def batch_detokenize(self, batch, remove_special_toks=False):
        batch = [
            self.detokenize(seq, remove_special_toks=remove_special_toks)
            for seq in batch
        ]
        return batch
    