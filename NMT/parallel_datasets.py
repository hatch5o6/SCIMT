from torch.utils.data import Dataset
import os
from tqdm import tqdm

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