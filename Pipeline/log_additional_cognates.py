import argparse
import os
import json
from tqdm import tqdm

"""
Logs the number of cognates in add_train, val, and test in the SC cfg files.
i.e. these are the cognates from CogNet and EtymDB only.
"""

def log(SC_cfg):
    langs = {}
    for f in tqdm(os.listdir(SC_cfg)):
        assert f.endswith(".cfg")
        lang = f.split(".cfg")[0]
        assert lang != ""
        assert lang not in langs
        
        f_path = os.path.join(SC_cfg, f)

        (add_train_src_f, 
         add_train_tgt_f, 
         val_src_f, 
         val_tgt_f, 
         test_src_f, 
         test_tgt_f) = read_cfg(f_path)
        
        add_train = read_parallel_f(add_train_src_f, add_train_tgt_f)
        val = read_parallel_f(val_src_f, val_tgt_f)
        test = read_parallel_f(test_src_f, test_tgt_f)

        langs[lang] = {
            "add_train": len(add_train),
            "val": len(val),
            "test": len(test),
            "total": len(add_train) + len(val) + len(test)
        }
    print("CogNet and EtymDB Cognates:")
    print(json.dumps(langs, ensure_ascii=False, indent=2))
        

def read_parallel_f(f1, f2):
    if os.path.exists(f1):
        assert os.path.exists(f2)
    else:
        assert not os.path.exists(f2)
        assert f1 == "null"
        assert f2 == "null"
    
    if os.path.exists(f2):
        assert os.path.exists(f1)
    else:
        assert not os.path.exists(f1)
        assert f1 == "null"
        assert f2 == "null"

    if f1 == "null":
        pairs = []
    else:
        with open(f1) as inf:
            lines1 = [l.strip() for l in inf.readlines()]
        with open(f2) as inf:
            lines2 = [l.strip() for l in inf.readlines()]
        assert len(lines1) == len(lines2)
        pairs = list(zip(lines1, lines2))

    return pairs

def read_cfg(f):
    add_train_src = None
    add_train_tgt = None
    val_src = None
    val_tgt = None
    test_src = None
    test_tgt = None
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    for line in lines:
        if line.strip().startswith("#"):
            continue

        if line.startswith("ADDITIONAL_TRAIN_COGNATES_SRC="):
            assert add_train_src == None
            add_train_src = line.split("ADDITIONAL_TRAIN_COGNATES_SRC=")[1]
        elif line.startswith("ADDITIONAL_TRAIN_COGNATES_TGT="):
            assert add_train_tgt == None
            add_train_tgt = line.split("ADDITIONAL_TRAIN_COGNATES_TGT=")[1]
        elif line.startswith("VAL_COGNATES_SRC="):
            assert val_src == None
            val_src = line.split("VAL_COGNATES_SRC=")[1]
        elif line.startswith("VAL_COGNATES_TGT="):
            assert val_tgt == None
            val_tgt = line.split('VAL_COGNATES_TGT=')[1]
        elif line.startswith("TEST_COGNATES_SRC="):
            assert test_src == None
            test_src = line.split("TEST_COGNATES_SRC=")[1]
        elif line.startswith("TEST_COGNATES_TGT="):
            assert test_tgt == None
            test_tgt = line.split("TEST_COGNATES_TGT=")[1]
    
    return (add_train_src, 
            add_train_tgt, 
            val_src, 
            val_tgt, 
            test_src, 
            test_tgt)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SC_cfg", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    log(args.SC_cfg)