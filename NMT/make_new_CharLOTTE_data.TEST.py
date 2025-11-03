import os
import csv
from assert_no_data_overlap import assert_no_overlap

NEW_DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN"
NEW_DATA_STORAGE = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data"

# NEW_DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/arabic"
# NEW_DATA_STORAGE = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_arabic"

def main():
    # Test that lang pairs in both folders match accordingly -- should be almost the same except NEW_DATA_STORAGE shouldn't have en-en and hi-hi
    assert os.listdir(NEW_DATA_DIR) == sorted(os.listdir(NEW_DATA_STORAGE) + ["en-en", "hi-hi"])
    # assert os.listdir(NEW_DATA_DIR) == sorted(os.listdir(NEW_DATA_STORAGE)) # if doing arabic
    # redundant test
    for d in os.listdir(NEW_DATA_STORAGE):
        assert d in os.listdir(NEW_DATA_DIR)
    

    for d in os.listdir(NEW_DATA_DIR):
        print(f"############################# {d} ##################################")
        csv_d = os.path.join(NEW_DATA_DIR, d)
        text_d = os.path.join(NEW_DATA_STORAGE, d)

        assert os.listdir(csv_d) == ["test.csv", "train.csv", "val.csv"]
        src_lang, tgt_lang = tuple([lang.strip() for lang in d.split("-")])
        assert src_lang != ""
        assert tgt_lang != ""        

        if d in ["en-en", "hi-hi"]:
            assert not os.path.exists(text_d)
            print(f"Asserted that storage for {d} ({text_d}) does not exists, as it shouldn't")
        
            train_csv_f = os.path.join(csv_d, "train.csv")
            val_csv_f = os.path.join(csv_d, "val.csv")
            test_csv_f = os.path.join(csv_d, "test.csv")

            train_src, train_tgt, train_src_path, train_tgt_path = get_paths_from_csv(train_csv_f)
            val_src, val_tgt, val_src_path, val_tgt_path = get_paths_from_csv(val_csv_f)
            test_src, test_tgt, test_src_path, test_tgt_path = get_paths_from_csv(test_csv_f)

            for fp in [train_src_path, train_tgt_path, val_src_path, val_tgt_path, test_src_path, test_tgt_path]:
                assert os.path.exists(fp)

            assert train_src == val_src == test_src == src_lang
            assert train_tgt == val_tgt == test_tgt == tgt_lang
            
            if d == "en-en":
                steal_from = "fr-en"
                steal_lang = "en"
            elif d == "hi-hi":
                steal_from = "bn-hi"
                steal_lang = "hi"
            
            assert train_src_path == train_tgt_path == os.path.join(NEW_DATA_STORAGE, steal_from, f"train.{steal_lang}.txt")
            assert val_src_path == val_tgt_path == os.path.join(NEW_DATA_STORAGE, steal_from, f"val.{steal_lang}.txt")
            assert test_src_path == test_tgt_path == os.path.join(NEW_DATA_STORAGE, steal_from, f"test.{steal_lang}.txt")
            
            print(f"CSV_D {d}, {csv_d} passed :)")
        else:
            assert os.path.exists(text_d)
            print(f"Asserted that storage for {d} ({text_d}) exists :)")

            csv_d_paths, csv_d_file_names, csv_d_src, csv_d_tgt, csv_files_parent_dir = get_all_paths_from_csv_d(csv_d)
            assert csv_files_parent_dir == text_d
            assert csv_d_src == src_lang
            assert csv_d_tgt == tgt_lang

            TEST_FILE_NAMES = [
                f"train.{src_lang}.txt",
                f"train.{tgt_lang}.txt",
                "train.notes",

                f"val.{src_lang}.txt",
                f"val.{tgt_lang}.txt",
                "val.notes",

                f"test.{src_lang}.txt",
                f"test.{tgt_lang}.txt",
                "test.notes",
            ]

            assert sorted(csv_d_file_names) == sorted(os.listdir(text_d)) == sorted(TEST_FILE_NAMES)

            TEST_FILE_PATHS = [
                os.path.join(NEW_DATA_STORAGE, f"{src_lang}-{tgt_lang}", f)
                for f in TEST_FILE_NAMES
            ]
            text_d_paths = [
                os.path.join(text_d, f)
                for f in os.listdir(text_d)
            ]

            assert sorted(TEST_FILE_PATHS) == sorted(text_d_paths) == sorted(csv_d_paths)
            #TODO I think we've confirmed the contents of the csv files and that they're correct and correctly point to NEW_DATA_STORAGE
            # Review code to confirm.
                # done
            # Then when that's done, need to confirm contents of the text data in NEW_DATA_STORAGE 
                # Make sure the set up for each language scenario makes sense.
                # make sure that there's no overlap between the train, val, test
                    # done
                # make sure that everything in train, val, test is respectively comes from the files mentioned in the .notes files
                    # so read in the content of train, val, test and read the original files and make sure that what's in train, val, test is contained in the original files.
                    # done
            
            print("\nASSERTING DATA IN TEXT D .txt FILES IS ALSO IN THE ORIGINAL DATA FROM .notes FILES")
            assert_content_of_text_d_is_in_notes(text_d, src_lang, tgt_lang, "train")
            assert_content_of_text_d_is_in_notes(text_d, src_lang, tgt_lang, "val")
            assert_content_of_text_d_is_in_notes(text_d, src_lang, tgt_lang, "test")

            print("\nASSERTING NO OVERLAP BETWEEN TRAIN, VAL, TEST")
            train_pairs = get_div_pairs(text_d, src_lang, tgt_lang, "train")
            val_pairs = get_div_pairs(text_d, src_lang, tgt_lang, "val")
            test_pairs = get_div_pairs(text_d, src_lang, tgt_lang, "test")
            overlap_passed, overlap_results = assert_no_overlap(train=train_pairs, dev=val_pairs, test=test_pairs, VERBOSE=True)
            for v in overlap_results.values():
                assert v == True
            assert overlap_passed == True
            print("\tThere is no overlap :)")


def get_div_pairs(text_d, src_lang, tgt_lang, div):
    assert div in ["train", "val", "test"]
    src_f = os.path.join(text_d, f"{div}.{src_lang}.txt")
    tgt_f = os.path.join(text_d, f"{div}.{tgt_lang}.txt")
    src_lines = read_f(src_f)
    tgt_lines = read_f(tgt_f)
    pairs = list(zip(src_lines, tgt_lines))
    return pairs


def assert_content_of_text_d_is_in_notes(text_d, src_lang, tgt_lang, div):
    assert div in ["train", "val", "test"]
    src_f = os.path.join(text_d, f"{div}.{src_lang}.txt")
    tgt_f = os.path.join(text_d, f"{div}.{tgt_lang}.txt")
    notes_f = os.path.join(text_d, f"{div}.notes")
    print("ASSERTING data pairs are in notes pairs:")
    print(f"\t  src_f: `{src_f}`")
    print(f"\t  tgt_f: `{tgt_f}`")
    print(f"\tnotes_f: `{notes_f}`")

    src_lines = read_f(src_f)
    tgt_lines = read_f(tgt_f)
    assert len(src_lines) == len(tgt_lines)
    data_pairs = list(zip(src_lines, tgt_lines))
    notes_pairs = read_notes(notes_f)

    for p in data_pairs:
        assert p in notes_pairs
        assert notes_pairs[p] > 0
        notes_pairs[p] -= 1
    
    for p, ct in notes_pairs.items():
        assert ct >= 0
    print(f"\tpassed :)")
    

def read_f(f):
    with open(f) as inf:
        content = [l.strip() for l in inf.readlines()]
    return content

def read_notes(f):
    with open(f) as inf:
        lines = [l.rstrip() for l in inf.readlines()]
    
    pairs = {}
    for l, line in enumerate(lines):
        if l == 0:
            assert line == "PAIRS CAME FROM THE FOLLOWING FILES:"
        elif l == len(lines) - 2:
            assert line == ""
        elif l == len(lines) -1 :
            assert line.startswith("2025-")
        else:
            assert line.startswith("\t- (`")
            tuple_str = line[3:]
            assert tuple_str.startswith("(`")
            assert tuple_str.endswith("`)")
            path_pair = tuple_str[1:-1]
            src_path, tgt_path = tuple(path_pair.split(", "))
            assert src_path.startswith("`") and src_path.endswith("`")
            assert tgt_path.startswith("`") and tgt_path.endswith("`")
            src_path = src_path[1:-1]
            tgt_path = tgt_path[1:-1]
            src_lines = read_f(src_path)
            tgt_lines = read_f(tgt_path)
            assert len(src_lines) == len(tgt_lines)
            for p in list(zip(src_lines, tgt_lines)):
                if p not in pairs:
                    pairs[p] = 0
                pairs[p] += 1
            
    return pairs

def get_all_paths_from_csv_d(csv_d):
    SET_SRC_LANG = None
    SET_TGT_LANG = None
    DIR = None
    paths = []
    for i, csv_f in enumerate(os.listdir(csv_d)):
        assert csv_f in ["test.csv", "train.csv", "val.csv"]

        csv_path = os.path.join(csv_d, csv_f)
        src_lang, tgt_lang, src_path, tgt_path = get_paths_from_csv(csv_path)
        
        if i == 0:
            assert SET_SRC_LANG == None
            assert SET_TGT_LANG == None
            assert DIR == None
            SET_SRC_LANG = src_lang
            SET_TGT_LANG = tgt_lang
            DIR = get_parent_dir(src_path)
            
        
        assert src_lang == SET_SRC_LANG
        assert tgt_lang == SET_TGT_LANG

        if csv_f == "test.csv":
            assert src_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/test.{SET_SRC_LANG}.txt")
            assert tgt_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/test.{SET_TGT_LANG}.txt")
        elif csv_f == "val.csv":
            assert src_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/val.{SET_SRC_LANG}.txt")
            assert tgt_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/val.{SET_TGT_LANG}.txt")
        elif csv_f == "train.csv":
            assert src_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/train.{SET_SRC_LANG}.txt")
            assert tgt_path == os.path.join(NEW_DATA_STORAGE, f"{SET_SRC_LANG}-{SET_TGT_LANG}/train.{SET_TGT_LANG}.txt")

        assert get_parent_dir(src_path) == get_parent_dir(tgt_path) == DIR
        paths.append(src_path)
        paths.append(tgt_path)
    paths.append(os.path.join(DIR, "train.notes"))
    paths.append(os.path.join(DIR, "val.notes"))
    paths.append(os.path.join(DIR, "test.notes"))
    file_names = [
        f.split("/")[-1]
        for f in paths
    ]
    return paths, file_names, SET_SRC_LANG, SET_TGT_LANG, DIR


def get_parent_dir(f):
    return "/".join(f.split("/")[:-1])

def get_paths_from_csv(csv_f):
    with open(csv_f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    assert len(rows) == 2
    assert rows[0] == ("src_lang", "tgt_lang", "src_path", "tgt_path")
    src_lang, tgt_lang, src_path, tgt_path = rows[1]
    return src_lang, tgt_lang, src_path, tgt_path


if __name__ == "__main__":
    main()