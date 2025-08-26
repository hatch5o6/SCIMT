from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import random

########### SC_ALIGNED_MULTILINGUAL_DATASET ###############

class SCAlignedMultilingualDataset(Dataset):
    def __init__(
        self,
        data_csv=None,
        sc_data_csv=None,
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

        print("SCAlignedMultilingualDataset READING CSV", "\n\t-DATA_CSV:", data_csv, "\n\t-SC_DATA_CSV:", sc_data_csv, "\n\t-SC_MODEL_ID:", sc_model_id)
        (src_tags, src_lines, sc_lines, 
         tgt_tags, tgt_lines, 
         FR_SRC_PATHS, SC_SRC_PATHS, FR_TGT_PATHS) = self.read_csvs(
            f=data_csv, 
            sc_f=sc_data_csv,
            sc_model_id=sc_model_id,
            upsample=upsample
        )

        self.src_paths = FR_SRC_PATHS
        self.sc_paths = SC_SRC_PATHS
        self.tgt_paths = FR_TGT_PATHS
    
        # set Nones in sc_lines to string <NONE>
        sc_lines_wo_nonetype = []
        for line in sc_lines:
            if line is None:
                sc_lines_wo_nonetype.append("<NONE>")
            else:
                sc_lines_wo_nonetype.append(line)
        sc_lines = sc_lines_wo_nonetype
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("parallel_datsets_sc, sc_lines:")
        # print(sc_lines)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        assert len(src_tags) == len(src_lines) == len(sc_lines) == len(tgt_tags) == len(tgt_lines)
        pairs = list(zip(src_tags, src_lines, sc_lines, tgt_tags, tgt_lines))
        print("SCAlignedMultilingualDataset TOTAL PAIRS:", len(pairs))
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
            # for item in data:
            #     assert isinstance(item, tuple)
            #     assert len(item) == 2
            #     assert isinstance(item[0], str)
            #     assert isinstance(item[1], str)

            #     if item not in track_unique:
            #         unique_data.append(item)
            #     track_unique.add(item)
            for fr_line, sc_line, tg_line in tqdm(data):
                assert isinstance(fr_line, str)
                assert isinstance(sc_line, str) or sc_line is None
                assert isinstance(tg_line, str)

                if (fr_line, tg_line) not in track_unique:
                    unique_data.append((fr_line, sc_line, tg_line))
                track_unique.add((fr_line, tg_line))
                
            # assert sorted(unique_data) == sorted(list(set(data)))
            print("\tAfter unique:", len(unique_data))
            data_by_pairs[lang_pair] = unique_data
        return data_by_pairs

    def read_data_csv(self, f, sc_model_id=None):
        with open(f, newline='') as inf:
            rows = [row for row in csv.reader(inf)]
        header = rows[0]
        assert header == ["src_lang", "tgt_lang", "src_path", "tgt_path"]

        data_by_pairs = {}
        rows = [tuple(row) for row in rows[1:]]
        SRC_PATHS = []
        TGT_PATHS = []
        SC_PAIRS = {}
        for src_lang, tgt_lang, src_path, tgt_path in rows:
            assert "SC_{SC_MODEL_ID}" not in tgt_path

            if self.limit_src_langs != None and src_lang not in self.limit_src_langs:
                continue
            if self.limit_tgt_langs != None and tgt_lang not in self.limit_tgt_langs:
                continue

            SRC_PATHS.append(src_path)
            TGT_PATHS.append(tgt_path)

            # Ensure that if a pair is an SC path (or not an SC path), 
            #   that every time there is a path with that pair it is also an SC path (or not an SC path)
            # e.g. ensure that if fr-en is an SC path, it is always an SC path
            # e.g. ensure that if mfe-en is not an SC path, is NEVER an SC path
            pair = (src_lang, tgt_lang)
            if pair not in SC_PAIRS:
                SC_PAIRS[pair] = None
            if "SC_{SC_MODEL_ID}" in src_path:
                assert SC_PAIRS[pair] in [None, True]
                SC_PAIRS[pair] = True
            else:
                assert SC_PAIRS[pair] in [None, False]
                SC_PAIRS[pair] = False
            assert SC_PAIRS[pair] is not None

            if pair not in data_by_pairs:
                data_by_pairs[pair] = []

            if "SC_{SC_MODEL_ID}" in src_path:
                assert SC_PAIRS[pair] == True
                assert sc_model_id is not None
                src_path = src_path.replace("{SC_MODEL_ID}", sc_model_id)
            else:
                assert SC_PAIRS[pair] == False

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
            if key not in self.raw_lengths:
                self.raw_lengths[key] = len(parallel_data)
            else:
                assert self.raw_lengths[key] == len(parallel_data)
            data_by_pairs[pair] += parallel_data
        return data_by_pairs, SRC_PATHS, TGT_PATHS, SC_PAIRS


    def synthesize_data_by_pairs(self, fr_data_by_pairs, sc_data_by_pairs, SC_PAIRS):
        # Assert they have the same pairs
        assert sorted(list(fr_data_by_pairs.keys())) == sorted(list(sc_data_by_pairs.keys()))
        synth_data_by_pairs = {}
        for pair, fr_data in fr_data_by_pairs.items():
            assert pair not in synth_data_by_pairs
            synth_data_by_pairs[pair] = []

            sc_data = sc_data_by_pairs[pair]
            assert len(fr_data) == len(sc_data)
            for i in range(len(fr_data)):
                fr_src, fr_tgt = fr_data[i]
                sc_src, sc_tgt = sc_data[i]
                assert fr_tgt == sc_tgt
                assert pair in SC_PAIRS
                # If the pair is not an SC Pair, then sc_src should be the same as fr_src, and we're not doing sound correspondence, and we don't need the sc_src to tokenize fr_src
                # print("SC_PAIRS")
                # print(SC_PAIRS)
                if SC_PAIRS[pair] == False:
                    assert sc_src == fr_src
                    sc_src = None
                elif SC_PAIRS[pair] == None:
                    assert sc_src == None
                else:
                    assert SC_PAIRS[pair] == True
                synth_data_by_pairs[pair].append((fr_src, sc_src, fr_tgt))
        return synth_data_by_pairs


    def read_csvs(self, f, sc_f, sc_model_id, upsample=False):        
        print("read_csvs", f, sc_f, sc_model_id)
        fr_data_by_pairs, FR_SRC_PATHS, FR_TGT_PATHS, FR_SC_PAIRS = self.read_data_csv(f)
        for scp, v in FR_SC_PAIRS.items():
            assert v == False

        if sc_f is not None:
            sc_data_by_pairs, SC_SRC_PATHS, SC_TGT_PATHS, SC_PAIRS = self.read_data_csv(sc_f, sc_model_id=sc_model_id)
            print("\nread sc_f with read_data_csv - SC_PAIRS should not have None")
        else:
            # If sc_f is None, then make a dummy sc_data_by_pairs with Nones for all the src_lines
            SC_PAIRS = {}
            sc_data_by_pairs = {}
            for lpair, lpair_data in fr_data_by_pairs.items():
                SC_PAIRS[lpair] = None
                assert lpair not in sc_data_by_pairs
                sc_data_by_pairs[lpair] = [
                    (None, ltgt_line) 
                    for lsrc_line, ltgt_line
                    in lpair_data
                ]
            SC_SRC_PATHS = [None for lpath in FR_SRC_PATHS]
            SC_TGT_PATHS = [lpath for lpath in FR_TGT_PATHS]
            print("created dummy sc data, SC_PAIRS should all be None")

        assert FR_TGT_PATHS == SC_TGT_PATHS
        assert FR_SRC_PATHS != SC_SRC_PATHS
        assert None not in FR_SRC_PATHS
        if None in SC_SRC_PATHS:
            for x in SC_SRC_PATHS:
                assert x == None
        else:
            assert None not in FR_SRC_PATHS
            assert None not in SC_SRC_PATHS
            assert sorted(FR_SRC_PATHS) != sorted(SC_SRC_PATHS)

        data_by_pairs = self.synthesize_data_by_pairs(fr_data_by_pairs, sc_data_by_pairs, SC_PAIRS)

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
        src_tags = []
        src_lines = []
        sc_lines = []
        tgt_tags = []
        tgt_lines = []
        for pair, pair_data in data_by_pairs.items():
            src, tgt = pair
            raw_data_size += len(pair_data)
            if upsample:
                print(f"UPSAMPLING {pair} ({len(pair_data)}) TO", MAX_SIZE)
                pair_data = self.upsample_data(pair_data, MAX_SIZE)
            upsampled_size += len(pair_data)
            for src_line, sc_line, tgt_line in pair_data:
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if sc_line is not None:
                    sc_line = sc_line.strip()

                src_tags.append(src)
                src_lines.append(src_line)
                sc_lines.append(sc_line)

                tgt_tags.append(tgt)
                tgt_lines.append(tgt_line)
        
        assert upsampled_size == len(src_tags) == len(src_lines) == len(sc_lines) == len(tgt_tags) == len(tgt_lines)
        print("RAW DATA SIZE:", raw_data_size)
        print("UPSAMPLED SIZE:", upsampled_size)
        print("RETURNING FR SRC PATHS", FR_SRC_PATHS)
        print("RETURNING SC SRC PATHS", SC_SRC_PATHS)
        print("RETURNING TGT PATHS", FR_TGT_PATHS)

        return src_tags, src_lines, sc_lines, tgt_tags, tgt_lines, FR_SRC_PATHS, SC_SRC_PATHS, FR_TGT_PATHS



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

