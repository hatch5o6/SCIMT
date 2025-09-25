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
        sc_model_id=None,
        append_src_lang_tok=False,
        append_tgt_lang_tok=True,
        append_tgt_to_src=False,
        size=None,
        seed=525,
        upsample=False,
        shuffle=False,
        limit_src_langs=None,
        limit_tgt_langs=None,
        CAN_RETURN_ZERO=False
    ):
        print("random seed:", seed)
        random.seed(seed)

        self.append_src_lang_tok = append_src_lang_tok
        self.append_tgt_to_src = append_tgt_to_src
        self.append_tgt_lang_tok = append_tgt_lang_tok
        self.shuffle = shuffle
        self.CAN_RETURN_ZERO = CAN_RETURN_ZERO
        self.lengths = {}
        self.raw_lengths = {}

        self.limit_src_langs = limit_src_langs
        if self.limit_src_langs != None:
            assert isinstance(self.limit_src_langs, list)
        self.limit_tgt_langs = limit_tgt_langs
        if self.limit_tgt_langs != None:
            assert isinstance(self.limit_tgt_langs, list)

        print("MultilingualDataset READING CSV", data_csv)
        src_lines, tgt_lines, SRC_PATHS, TGT_PATHS = self.read_csv(data_csv, sc_model_id=sc_model_id, upsample=upsample)

        self.src_paths = SRC_PATHS
        self.tgt_paths = TGT_PATHS
    
        assert len(src_lines) == len(tgt_lines)
        pairs = list(zip(src_lines, tgt_lines))
        print("MultilingualDataset TOTAL PAIRS:", len(pairs))
        if self.shuffle:
            random.shuffle(pairs)
        self.pairs = pairs
    
    def make_data_by_pairs_unique(self, data_by_pairs):
        for lang_pair, data in data_by_pairs.items():
            print("Making unique:", lang_pair)
            assert isinstance(data, list)
            print("\tBefore unique:", len(data))
            unique_data = []
            track_unique = set()
            for item in data:
                assert isinstance(item, tuple)
                assert len(item) == 2
                assert isinstance(item[0], str)
                assert isinstance(item[1], str)

                if item not in track_unique:
                    unique_data.append(item)
                track_unique.add(item)
            assert sorted(unique_data) == sorted(list(set(data)))
            print("\tAfter unique:", len(unique_data))
            data_by_pairs[lang_pair] = unique_data
        return data_by_pairs

    def read_csv(self, f, sc_model_id=None, upsample=False):
        # if sc_model_id != None: # Not sure why I wrote these lines.
        #     assert f.endswith("/train.no_overlap_v1.csv") or f.endswith("/val.no_overlap_v1.csv")
        # else:
        #     assert f.endswith("/test.csv") or f.endswith("/inference.csv")

        with open(f, newline='') as inf:
            rows = [row for row in csv.reader(inf)]
        header = rows[0]
        assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]

        data_by_pairs = {}
        rows = [tuple(row) for row in rows[1:]]
        SRC_PATHS = []
        TGT_PATHS = []
        for src_lang, tgt_lang, src_path, tgt_path in rows:
            assert "SC_{SC_MODEL_ID}" not in tgt_path
            if sc_model_id == None:
                assert "SC_{SC_MODEL_ID}" not in src_path
            
            if "SC_{SC_MODEL_ID}" in src_path:
                assert sc_model_id != None
                src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)

            if self.limit_src_langs != None and src_lang not in self.limit_src_langs:
                continue
            if self.limit_tgt_langs != None and tgt_lang not in self.limit_tgt_langs:
                continue

            SRC_PATHS.append(src_path)
            TGT_PATHS.append(tgt_path)

            pair = (src_lang, tgt_lang)
            if pair not in data_by_pairs:
                data_by_pairs[pair] = []

            lang_src_lines = self.read_file(src_path)
            lang_tgt_lines = self.read_file(tgt_path)

            if self.append_src_lang_tok:
                lang_src_lines = [f"<{src_lang}>" + line for line in lang_src_lines]
            elif self.append_tgt_to_src:
                lang_src_lines = [f"<{tgt_lang}>" + line for line in lang_src_lines]
            if self.append_tgt_lang_tok:
                lang_tgt_lines = [f"<{tgt_lang}>" + line for line in lang_tgt_lines]

            parallel_data = list(zip(lang_src_lines, lang_tgt_lines))
            key = f"{src_lang}-{tgt_lang}, `{src_path}`, `{tgt_path}`"
            assert key not in self.raw_lengths
            self.raw_lengths[key] = len(parallel_data)
            data_by_pairs[pair] += parallel_data
        
        data_by_pairs = self.make_data_by_pairs_unique(data_by_pairs)
        
        for lang_pair, data in data_by_pairs.items():
            l1, l2 = lang_pair

            assert f"{l1}-{l2}" not in self.lengths
            self.lengths[f"{l1}-{l2}"] = len(data)

        MAX_SIZE = 0
        for pair, pair_data in data_by_pairs.items():
            if len(pair_data) > MAX_SIZE:
                MAX_SIZE = len(pair_data)
        if not self.CAN_RETURN_ZERO:
            assert MAX_SIZE > 0
        print("DATA MAX_SIZE =", MAX_SIZE)

        raw_data_size = 0
        upsampled_size = 0
        src_lines = []
        tgt_lines = []
        for pair, pair_data in data_by_pairs.items():
            raw_data_size += len(pair_data)
            if upsample:
                print(f"UPSAMPLING {pair} ({len(pair_data)}) TO", MAX_SIZE)
                pair_data = self.upsample_data(pair_data, MAX_SIZE)
            upsampled_size += len(pair_data)
            for src_line, tgt_line in pair_data:
                src_lines.append(src_line.strip())
                tgt_lines.append(tgt_line.strip())
        
        assert upsampled_size == len(src_lines) == len(tgt_lines)
        print("RAW DATA SIZE:", raw_data_size)
        print("UPSAMPLED SIZE:", upsampled_size)
        print("RETURNING SRC PATHS", SRC_PATHS)
        print("RETURNING TGT PATHS", TGT_PATHS)

        return src_lines, tgt_lines, SRC_PATHS, TGT_PATHS

    def upsample_data(self, data, final_size):
        assert final_size >= len(data)
        if self.shuffle:
            random.shuffle(data)
        while len(data) < final_size:
            data += data
        if len(data) > final_size:
            data = data[:final_size]
        return data

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
