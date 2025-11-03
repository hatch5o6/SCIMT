import csv
import os
import shutil
from assert_no_data_overlap import assert_no_overlap
from datetime import datetime
import json
from tqdm import tqdm

DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/PLAIN"
NEW_DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN"

# # if doing arabic
# DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/arabic"
# NEW_DATA_DIR = "/home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/arabic"

NEW_DATA_STORAGE = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data"

# # if doing arabic
# NEW_DATA_STORAGE = "/home/hatch5o6/nobackup/archive/data/CharLOTTE_arabic"

if os.path.exists(NEW_DATA_DIR):
    print("Removing", NEW_DATA_DIR)
    shutil.rmtree(NEW_DATA_DIR)
print("Creating", NEW_DATA_DIR)
os.mkdir(NEW_DATA_DIR)


if os.path.exists(NEW_DATA_STORAGE):
    print("DELETING", NEW_DATA_STORAGE)
    shutil.rmtree(NEW_DATA_STORAGE)
print("CREATING", NEW_DATA_STORAGE)
os.mkdir(NEW_DATA_STORAGE)

# comment out list items if doing arabic
SKIP_DIRS = [
    "mat-thewtgt",
    "mat-thewtgt_dev_test",
    "matsrc-thew",
    "matsrc-thew_dev_test",
    "matsrc-thewtgt",
    "matsrc-thewtgt_dev_test"
]

def main():
    for sd in SKIP_DIRS:
        assert sd in os.listdir(DATA_DIR)

    pairs = {}
    for d in os.listdir(DATA_DIR):
        if d in SKIP_DIRS: continue
        pair = d.split("_")[0]
        src, tgt = tuple([l.strip() for l in pair.split("-")])
        assert src != ""
        assert tgt != ""
        
        if (src, tgt) not in pairs:
            pairs[(src, tgt)] = []
        pairs[(src, tgt)].append(d)

    for src, tgt in tqdm(pairs):
        print(f"###################### SETTING UP CharLOTTE DATA FOR PAIR {src}-{tgt} #######################")

        pair_new_dir = os.path.join(NEW_DATA_DIR, f"{src}-{tgt}")
        assert not os.path.exists(pair_new_dir)
        os.mkdir(pair_new_dir)

        if (src, tgt) in [("en", "en"), ("hi", "hi")]:
            if (src, tgt) == ("en", "en"):
                assert pairs[(src, tgt)] == ["en-en_CCMatrix", "en-en_FLORES+"]
                print(f"PAIR {(src, tgt)} has dirs {['en-en_CCMatrix', 'en-en_FLORES+']} as it should :)")
                language = "en"
                steal_from = os.path.join(NEW_DATA_STORAGE, "fr-en")

            elif (src, tgt) == ("hi", "hi"):
                assert pairs[(src, tgt)] == ["hi-hi_NLLB"]
                print(f"PAIR {(src, tgt)} has dirs {['hi-hi_NLLB']} as it should :)")
                language = "hi"
                steal_from = os.path.join(NEW_DATA_STORAGE, "bn-hi")

            train_file = os.path.join(steal_from, f"train.{language}.txt")
            train_data = [(language, language, train_file, train_file)]
            write_csv(train_data, os.path.join(pair_new_dir, "train.csv"))

            val_file = os.path.join(steal_from, f"val.{language}.txt")
            val_data = [(language, language, val_file, val_file)]
            write_csv(val_data, os.path.join(pair_new_dir, "val.csv"))

            test_file = os.path.join(steal_from, f"test.{language}.txt")
            test_data = [(language, language, test_file, test_file)]
            write_csv(test_data, os.path.join(pair_new_dir, "test.csv"))

            # for d in pairs[(src, tgt)]:
            #     d_path = os.path.join(DATA_DIR, d)
            #     for f in os.listdir(d_path):
            #         assert f == "extra_tokenizer_data.csv"
            #         f_path = os.path.join(d_path, f)
            #         data = read_csv(f_path)

            #         new_data = []
            #         for data_src, data_tgt, src_path, tgt_path in data:
            #             assert (data_src, data_tgt) == (src, tgt)
            #             EXT = src_path.split(".")[-1]
            #             assert tgt_path.endswith(f".{EXT}")

            #             if src_path.endswith(f".no_overlap_v1.{EXT}"):
            #                 assert tgt_path.endswith(f".no_overlap_v1.{EXT}")
            #                 src_path = src_path[:-len(f".no_overlap_v1.{EXT}")] + f".{EXT}"
            #                 tgt_path = tgt_path[:-len(f".no_overlap_v1.{EXT}")] + f".{EXT}"

            #                 assert os.path.exists(src_path)
            #                 assert os.path.exists(tgt_path)
            #             new_data.append([data_src, data_tgt, src_path, tgt_path])
                    
            #         new_f = os.path.join(pair_new_dir, d + "_" + f)
            #         assert not os.path.exists(new_f)
            #         write_csv(new_data, new_f)
                    
        else:
            assert pairs[(src, tgt)] == [f"{src}-{tgt}", f"{src}-{tgt}_dev_test"]
            print(f"PAIR {(src, tgt)} has dirs {[f'{src}-{tgt}', f'{src}-{tgt}_dev_test']} as it should :)")
            train_csv = os.path.join(DATA_DIR, f"{src}-{tgt}/train.csv")
            train_no_csv = os.path.join(DATA_DIR, f"{src}-{tgt}/train.no_overlap_v1.csv")
            assert os.path.exists(train_no_csv)
            if os.path.exists(train_csv):
                assert_csv_and_no_csv_differ_the_right_way(train_csv, train_no_csv)
            else:
                train_csv = create_csv_from_no_csv(no_csv=train_no_csv)

            val_csv = os.path.join(DATA_DIR, f"{src}-{tgt}_dev_test/val.csv")
            val_no_csv = os.path.join(DATA_DIR, f"{src}-{tgt}_dev_test/val.no_overlap_v1.csv")
            assert os.path.exists(val_no_csv)
            if os.path.exists(val_csv):
                assert_csv_and_no_csv_differ_the_right_way(val_csv, val_no_csv)
            else:
                val_csv = create_csv_from_no_csv(no_csv=val_no_csv)

            test_csv = os.path.join(DATA_DIR, f"{src}-{tgt}_dev_test/test.csv")
            assert os.path.exists(test_csv)

            
            pair_data_storage = os.path.join(NEW_DATA_STORAGE, f"{src}-{tgt}")
            assert not os.path.exists(pair_data_storage)
            os.mkdir(pair_data_storage)

            train, train_paths, train_src, train_tgt = read_csv_text_data(train_csv)
            val, val_paths, val_src, val_tgt = read_csv_text_data(val_csv)
            test, test_paths, test_src, test_tgt = read_csv_text_data(test_csv)

            assert train_src == val_src == test_src == src
            assert train_tgt == val_tgt == test_tgt == tgt

            new_train, new_val, new_test = remove_overlap(train, val, test)
            assert new_test == test # these shouldn't be different

            passed, results = assert_no_overlap(
                train=new_train,
                dev=new_val,
                test=new_test,
                VERBOSE=True
            )
            print("IS THERE OVERLAP?")
            print(json.dumps(results, ensure_ascii=False, indent=2))
            assert passed == True

            print("Writing data")
            write_pairs(
                pairs=new_train,
                data_paths=train_paths,
                pair_data_storage=pair_data_storage,
                pair_new_dir=pair_new_dir,
                src_lang=src,
                tgt_lang=tgt,
                div="train"
            )

            write_pairs(
                pairs=new_val,
                data_paths=val_paths,
                pair_data_storage=pair_data_storage,
                pair_new_dir=pair_new_dir,
                src_lang=src,
                tgt_lang=tgt,
                div="val"
            )

            write_pairs(
                pairs=new_test,
                data_paths=test_paths,
                pair_data_storage=pair_data_storage,
                pair_new_dir=pair_new_dir,
                src_lang=src,
                tgt_lang=tgt,
                div="test"
            )


def write_pairs(pairs, data_paths, pair_data_storage, pair_new_dir, src_lang, tgt_lang, div):
    assert div in ["train", "val", "test"]
    save_src = os.path.join(pair_data_storage, f"{div}.{src_lang}.txt")
    save_tgt = os.path.join(pair_data_storage, f"{div}.{tgt_lang}.txt")
    notes_path = os.path.join(pair_data_storage, f"{div}.notes")

    assert not os.path.exists(save_src)
    assert not os.path.exists(save_tgt)
    assert not os.path.exists(notes_path)

    # Write text data
    with open(save_src, "w") as sf, open(save_tgt, "w") as tf:
        for src_line, tgt_line in pairs:
            sf.write(src_line.strip() + "\n")
            tf.write(tgt_line.strip() + "\n")
    # Write text data notes -- i.e. origins of the data
    with open(notes_path, "w") as outf:
        outf.write("PAIRS CAME FROM THE FOLLOWING FILES:\n")
        for src_path, tgt_path in data_paths:
            outf.write(f"\t- (`{src_path}`, `{tgt_path}`)\n")
        outf.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Write new data .csv files
    csv_path = os.path.join(pair_new_dir, f"{div}.csv")
    assert not os.path.exists(csv_path)
    csv_data = [(src_lang, tgt_lang, save_src, save_tgt)]
    write_csv(data=csv_data, f=csv_path)

def remove_overlap(train, val, test):
    print("REMOVING OVERLAP")
    print("\tTRAIN:", len(train))
    print("\t  VAL:", len(val))
    print("\t TEST:", len(test))

    val_src_set, val_tgt_set = get_src_tgt_sets(val)
    test_src_set, test_tgt_set = get_src_tgt_sets(test)

    # For every src or tgt sent in train that is also in val or test, remove it
    new_train_pairs = []
    for train_src_seg, train_tgt_seg in train:
        REMOVE = any([
            train_src_seg in val_src_set,
            train_src_seg in test_src_set,

            train_tgt_seg in val_tgt_set,
            train_tgt_seg in test_tgt_set
        ])
        if not REMOVE:
            new_train_pairs.append((train_src_seg, train_tgt_seg))
    
    # For every src or tgt sent in val that is also in test, remove it
    new_val_pairs = []
    for val_src_seg, val_tgt_seg in val:
        REMOVE = any([
            val_src_seg in test_src_set,

            val_tgt_seg in test_tgt_set
        ])
        if not REMOVE:
            new_val_pairs.append((val_src_seg, val_tgt_seg))
    
    print("\t--->")
    print("\tTRAIN:", len(new_train_pairs))
    print("\t  VAL:", len(new_val_pairs))
    print("\t TEST:", len(test))
    # There should now be no overlap at all between these sets being returned, not at the pair or segment level
    return new_train_pairs, new_val_pairs, test


def get_src_tgt_sets(pairs):
    src_set = set([src for src, tgt in pairs])
    tgt_set = set([tgt for src, tgt in pairs])
    return src_set, tgt_set

def read_csv_text_data(csv_file):
    rows = read_csv(csv_file)
    SET_SRC_LANG = None
    SET_TGT_LANG = None
    src_lines = []
    tgt_lines = []
    src_file_paths = []
    tgt_file_paths = []
    for i, (src_lang, tgt_lang, src_path, tgt_path) in enumerate(rows):
        if i == 0:
            assert SET_SRC_LANG == None
            assert SET_TGT_LANG == None
            SET_SRC_LANG = src_lang
            SET_TGT_LANG = tgt_lang
        
        assert src_lang == SET_SRC_LANG
        assert tgt_lang == SET_TGT_LANG

        src_lines += read_f(src_path)
        tgt_lines += read_f(tgt_path)
        assert len(src_lines) == len(tgt_lines)

        src_file_paths.append(src_path)
        tgt_file_paths.append(tgt_path)
    assert len(src_lines) == len(tgt_lines)

    pairs = list(zip(src_lines, tgt_lines))

    assert len(src_file_paths) == len(tgt_file_paths)
    path_pairs = list(zip(src_file_paths, tgt_file_paths))
    return pairs, path_pairs, SET_SRC_LANG, SET_TGT_LANG

def read_f(f):
    with open(f) as inf:
        content = [l.strip() for l in inf.readlines()]
    return content
    

def assert_csv_and_no_csv_differ_the_right_way(csv, no_csv):
    csv_data = read_csv(csv)
    no_csv_data = read_csv(no_csv)

    assert isinstance(csv_data, list)
    assert isinstance(no_csv_data, list)
    assert len(csv_data) == len(no_csv_data)

    for i in range(len(csv_data)):
        csv_row = csv_data[i]
        no_csv_row = no_csv_data[i]
        assert len(csv_row) == len(no_csv_row) == 4

        assert csv_row[0] == no_csv_row[0] # assert src langs are the same
        assert csv_row[1] == no_csv_row[1] # assert tgt langs are the same

        csv_src_path = csv_row[2]
        no_csv_src_path = no_csv_row[2]
        compare_paths_pair(csv_src_path, no_csv_src_path)
        
        csv_tgt_path = csv_row[3]
        no_csv_tgt_path = no_csv_row[3]
        compare_paths_pair(csv_tgt_path, no_csv_tgt_path)
    
    print(f"`{csv}` and `{no_csv}` differ in the way expected, with just .no_overlap_v1 extensions")


def compare_paths_pair(csv_path, no_csv_path):
    EXT = csv_path.split(".")[-1]
    assert no_csv_path.endswith(f".{EXT}")

    assert no_csv_path.endswith(f".no_overlap_v1.{EXT}")
    assert csv_path == no_csv_path[:-len(f".no_overlap_v1.{EXT}")] + f".{EXT}"


def create_csv_from_no_csv(no_csv):
    new_csv_name = no_csv.split("/")[-1]
    d = "/".join(no_csv.split("/")[:-1])
    EXT = new_csv_name.split(".")[-1]
    assert new_csv_name.endswith(f".no_overlap_v1.{EXT}")
    new_csv_name = new_csv_name[:-len(f".no_overlap_v1.{EXT}")] + f".{EXT}"
    new_csv_path = os.path.join(d, new_csv_name)
    assert not os.path.exists(new_csv_path)

    no_csv_data = read_csv(no_csv)

    with open(new_csv_path, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["src_lang", "tgt_lang", "src_path", "tgt_path"])
        for src_lang, tgt_lang, src_path, tgt_path in no_csv_data:
            SRC_EXT = src_path.split(".")[-1]
            TGT_EXT = tgt_path.split(".")[-1]
            if src_path.endswith(f".no_overlap_v1.{SRC_EXT}"):
                assert tgt_path.endswith(f".no_overlap_v1.{TGT_EXT}")
                src_path = src_path[:-len(f".no_overlap_v1.{SRC_EXT}")] + f".{SRC_EXT}"
                tgt_path = tgt_path[:-len(f".no_overlap_v1.{TGT_EXT}")] + f".{TGT_EXT}"

            writer.writerow([src_lang, tgt_lang, src_path, tgt_path])
    return new_csv_path


def write_csv(data, f):
    with open(f, "w", newline='') as outf:
        writer = csv.writer(outf)
        writer.writerow(["src_lang", "tgt_lang", "src_path", "tgt_path"])
        for src_lang, tgt_lang, src_path, tgt_path in data:
            writer.writerow([src_lang, tgt_lang, src_path, tgt_path])

def read_csv(f):
    with open(f, newline='') as inf:
        rows = [tuple(r) for r in csv.reader(inf)]
    assert rows[0] == ("src_lang", "tgt_lang", "src_path", "tgt_path")
    data = rows[1:]
    return data


if __name__ == "__main__":
    main()