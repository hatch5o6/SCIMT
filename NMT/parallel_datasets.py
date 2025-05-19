from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import random

class ParallelDataset(Dataset):
    def __init__(
            self,
            src_file_path=None, 
            tgt_file_path=None,
            size=None
        ):
        src_lines = self.read_file(src_file_path, size)
        tgt_lines = self.read_file(tgt_file_path, size)
        
        assert len(src_lines) == len(tgt_lines)
        self.pairs = list(zip(src_lines, tgt_lines))

    def read_file(self, f, size=None):
        if not size:
            print(f"Reading all lines from {f}")
            with open(f) as inf:
                lines = [line.strip() for line in inf.readlines()]
        else:
            print(f"Reading {size} lines from {f}")
            progress = tqdm(total=size)
            lines = []
            with open(f) as inf:
                line = inf.readline()
                while line and len(lines) <= size:
                    line = line.strip()
                    lines.append(line)
                    line = inf.readline()
                    progress.update(1)
        return lines

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]


class MultilingualDataset(Dataset):
    def __init__(
            self,
            data_csv=None,
            append_src_lang_tok=False,
            append_tgt_lang_tok=True,
            size=None,
            seed=525
        ):
        print("random seed:", seed)
        random.seed(seed)

        self.append_src_lang_tok = append_src_lang_tok
        self.append_tgt_lang_tok = append_tgt_lang_tok

        src_lines, tgt_lines = self.read_csv(data_csv)
        
        assert len(src_lines) == len(tgt_lines)
        pairs = list(zip(src_lines, tgt_lines))
        random.shuffle(pairs)
        self.pairs = pairs

    def read_csv(self, f):
        with open(f, newline='') as inf:
            rows = [row for row in csv.reader(inf)]
        header = rows[0]
        assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]

        src_lines = []
        tgt_lines = []
        rows = [tuple(row) for row in rows[1:]]
        for src_lang, tgt_lang, src_path, tgt_path in rows:
            lang_src_lines = self.read_file(src_path)
            lang_tgt_lines = self.read_file(tgt_path)
            if self.append_src_lang_tok:
                lang_src_lines = [f"<{src_lang}>" + line for line in lang_src_lines]
            if self.append_tgt_lang_tok:
                lang_tgt_lines = [f"<{tgt_lang}>" + line for line in lang_tgt_lines]

            src_lines += lang_src_lines
            tgt_lines += lang_tgt_lines

        return src_lines, tgt_lines

    def read_file(self, f, size=None):
        if not size:
            print(f"Reading all lines from {f}")
            with open(f) as inf:
                lines = [line.strip() for line in inf.readlines()]
        else:
            print(f"Reading {size} lines from {f}")
            progress = tqdm(total=size)
            lines = []
            with open(f) as inf:
                line = inf.readline()
                while line and len(lines) <= size:
                    line = line.strip()
                    lines.append(line)
                    line = inf.readline()
                    progress.update(1)
        return lines

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        return self.pairs[index]