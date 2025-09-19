import argparse
import os
import shutil
import csv
import copy
from datetime import datetime

"""
Provided the .cfgs in /home/hatch5o6/Cognate/code/Pipeline/cfg/SC, makes .csv files for parallel augmented and SC data.
"""

GT_SC =        "/home/hatch5o6/Cognate/code/NMT/data/TEST_SC"
GT_AUG_SC =    "/home/hatch5o6/Cognate/code/NMT/TEST_augmented_data/SC"
GT_AUG_PLAIN = "/home/hatch5o6/Cognate/code/NMT/TEST_augmented_data/PLAIN"

# GT_SC =        "/home/hatch5o6/Cognate/code/NMT/data/SC_jul14"
# GT_AUG_SC =    "/home/hatch5o6/Cognate/code/NMT/augmented_data_jul14/SC"
# GT_AUG_PLAIN = "/home/hatch5o6/Cognate/code/NMT/augmented_data_jul14/PLAIN"

def test_aug():
    pass

def test_sc():
    pass

def test_dirs(GT, HYP):
    print(f"COMPARING {HYP}")
    print(f"\tTO GT", GT)
    total = 0
    passed = 0
    for d in os.listdir(GT):
        gt_d = os.path.join(GT, d)
        hyp_d = os.path.join(HYP, d)
        for f in os.listdir(gt_d):
            total += 1
            gt_f = os.path.join(gt_d, f)
            hyp_f = os.path.join(hyp_d, f)
            
            with open(gt_f) as inf:
                gt_f_content = inf.read()
            with open(hyp_f) as inf:
                hyp_f_content = inf.read()
            if hyp_f_content == gt_f_content:
                passed += 1
                # print("\t\t---------------------")
                # print(f"\t\thyp:", hyp_f)
                # print(f"\t\tgt:", gt_f)
                # print("\t\tEQUAL!")
            else:
                print("\t\t---------------------")
                print(f"\t\thyp:", hyp_f)
                print(f"\t\tgt:", gt_f)
                print("\t\tNOT EQUAL :(")
    print("PASSED:", passed)
    print("TOTAL:", total)

def make_data(
    cfgs,
    nmt_data_dir,
    aug_out_dir,
    INSERT_NO_OVERLAP=None
):
    time = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    if os.path.exists(aug_out_dir):
        print("deleting", aug_out_dir)
        shutil.rmtree(aug_out_dir)
    print("creating", aug_out_dir)
    os.mkdir(aug_out_dir)
    aug_plain_dir = os.path.join(aug_out_dir, "PLAIN")
    os.mkdir(aug_plain_dir)
    aug_sc_dir = os.path.join(aug_out_dir, "SC")
    os.mkdir(aug_sc_dir)

    nmt_plain_dir = os.path.join(nmt_data_dir, "PLAIN")
    assert os.path.exists(nmt_plain_dir)
    nmt_sc_dir = os.path.join(nmt_data_dir, "SC")
    if os.path.exists(nmt_sc_dir):
        print("deleting", nmt_sc_dir)
        shutil.rmtree(nmt_sc_dir)
    print("creating", nmt_sc_dir)
    os.mkdir(nmt_sc_dir)

    for cfg in os.listdir(cfgs):
        if "en-djk" in cfg: continue
        cfg_path = os.path.join(cfgs, cfg)
        make_augmented_dir(
            cfg_file=cfg_path, 
            nmt_data_dir=nmt_data_dir, 
            aug_out_dir=aug_out_dir,
            INSERT_NO_OVERLAP=INSERT_NO_OVERLAP
        )

    if INSERT_NO_OVERLAP == None:
        print("TEST NMT SC")
        test_dirs(
            GT=GT_SC,
            HYP=nmt_sc_dir
        )
        print("TEST AUG PLAIN")
        test_dirs(
            GT=GT_AUG_PLAIN,
            HYP=aug_plain_dir
        )
        print("TEST AUG SC")
        test_dirs(
            GT=GT_AUG_SC,
            HYP=aug_sc_dir
        )

def make_SC(csv_file, NMT_SRC, AUG_SRC, TGT, COG_SRC, COG_TGT, sc_out_dir, IS_COG=False, DEV_TEST=False):
    csvfname_wext = csv_file.split("/")[-1]
    if DEV_TEST:
        if IS_COG == False:
            sc_csv_dir = f"SC_{COG_SRC}2{COG_TGT}-{TGT}_dev_test"
        else:
            sc_csv_dir = f"SC_{COG_SRC}2{COG_TGT}-{COG_TGT}_dev_test"
        sc_csv_dir_path = os.path.join(sc_out_dir, sc_csv_dir)
        if not os.path.exists(sc_csv_dir_path):
            print("creating", sc_csv_dir_path)
            os.mkdir(sc_csv_dir_path)
    else:
        if IS_COG == False:
            sc_csv_dir = f"SC_{COG_SRC}2{COG_TGT}-{TGT}"
        else:
            sc_csv_dir = f"SC_{COG_SRC}2{COG_TGT}-{COG_TGT}"
        sc_csv_dir_path = os.path.join(sc_out_dir, sc_csv_dir)
        if os.path.exists(sc_csv_dir_path):
            print("EXISTS:", sc_csv_dir_path)
        assert not os.path.exists(sc_csv_dir_path)
        os.mkdir(sc_csv_dir_path)

    sc_csv_path = os.path.join(sc_csv_dir_path, csvfname_wext)

    csv_header, csv_data = read_csv(csv_file)
    sc_csv_data = []
    for row in csv_data:
        if row[0] not in [NMT_SRC, AUG_SRC]:
            print("IN CSV FILE", csv_file)
            print(f"\t{row[0]} not in {[NMT_SRC, AUG_SRC]}")
        assert row[0] in [NMT_SRC, AUG_SRC]
        if IS_COG == False:
            assert row[1] == TGT
        else:
            assert row[1] in [NMT_SRC, AUG_SRC]

        if row[0] == COG_SRC:
            new_row = copy.deepcopy(row)
            src_path = new_row[2]
            ext = src_path.split(".")[-1]
            src_path = src_path[:-len(ext)] + f"SC_{{SC_MODEL_ID}}_{COG_SRC}2{COG_TGT}.{ext}"
            new_row[2] = src_path
            sc_csv_data.append(new_row)
        else:
            sc_csv_data.append(row)
    
    sc_csv_content = [csv_header] + sc_csv_data
    with open(sc_csv_path, "w", newline='') as outf:
        writer = csv.writer(outf)
        for r in sc_csv_content:
            writer.writerow(r)

def make_augmented_dir(cfg_file, nmt_data_dir, aug_out_dir, INSERT_NO_OVERLAP):
    # Makes augmented data and SC data files

    nmt_plain_dir = os.path.join(nmt_data_dir, "PLAIN")
    nmt_sc_dir = os.path.join(nmt_data_dir, "SC")
    aug_plain_dir = os.path.join(aug_out_dir, "PLAIN")
    aug_sc_dir = os.path.join(aug_out_dir, "SC")

    NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, COG_SRC, COG_TGT = read_config(cfg_file)
    print("NMT_SRC", NMT_SRC)
    print("NMT_TGT", NMT_TGT)
    print("AUG_SRC", AUG_SRC)
    print("AUG_TGT", AUG_TGT)
    print("COG_SRC", COG_SRC)
    print("COG_TGT", COG_TGT)
    assert NMT_TGT == AUG_TGT

    
    assert NMT_TGT == AUG_TGT

    if INSERT_NO_OVERLAP != None:
        tag = f".no_overlap_{INSERT_NO_OVERLAP}"
    else:
        tag = ""

    NMT_f = os.path.join(nmt_plain_dir, f"{NMT_SRC}-{NMT_TGT}", f"train{tag}.csv")
    NMT_val_f = os.path.join(nmt_plain_dir, f"{NMT_SRC}-{NMT_TGT}_dev_test", f"val{tag}.csv")
    NMT_test_f = os.path.join(nmt_plain_dir, f"{NMT_SRC}-{NMT_TGT}_dev_test", "test.csv")
    assert os.path.exists(NMT_f)
    assert os.path.exists(NMT_val_f)
    assert os.path.exists(NMT_test_f)

    if NMT_SRC == AUG_SRC == COG_SRC:
        make_SC(NMT_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir)
        make_SC(NMT_val_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, DEV_TEST=True)
        make_SC(NMT_test_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, DEV_TEST=True)
    else:
        # -TEST THIS- #
        COG_f = os.path.join(nmt_plain_dir, f"{COG_SRC}-{COG_TGT}", f"train{tag}.csv")
        COG_val_f = os.path.join(nmt_plain_dir, f"{COG_SRC}-{COG_TGT}_dev_test", f"val{tag}.csv")
        COG_test_f = os.path.join(nmt_plain_dir, f"{COG_SRC}-{COG_TGT}_dev_test", "test.csv")
        print("COG_f:", COG_f)
        assert os.path.exists(COG_f)
        assert os.path.exists(COG_val_f)
        assert os.path.exists(COG_test_f)

        make_SC(COG_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, IS_COG=True)
        make_SC(COG_val_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, IS_COG=True, DEV_TEST=True)
        make_SC(COG_test_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, IS_COG=True, DEV_TEST=True)
        # --- #



        AUG_f = os.path.join(nmt_plain_dir, f"{AUG_SRC}-{AUG_TGT}", f"train{tag}.csv")
        AUG_val_f = os.path.join(nmt_plain_dir, f"{AUG_SRC}-{AUG_TGT}_dev_test", f"val{tag}.csv")
        AUG_test_f = os.path.join(nmt_plain_dir, f"{AUG_SRC}-{AUG_TGT}_dev_test", "test.csv")
        assert os.path.exists(AUG_f)
        assert os.path.exists(AUG_val_f)
        assert os.path.exists(AUG_test_f)

        make_SC(AUG_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir)
        make_SC(AUG_val_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, DEV_TEST=True)
        make_SC(AUG_test_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=nmt_sc_dir, DEV_TEST=True)

        aug_plain_langpair_dir = os.path.join(aug_plain_dir, f"{NMT_SRC}-{NMT_TGT}")
        assert not os.path.exists(aug_plain_langpair_dir)
        os.mkdir(aug_plain_langpair_dir)
        COMBINED_f = os.path.join(aug_plain_langpair_dir, f"train{tag}.csv")
        combine_csvs(NMT_f, AUG_f, out_csv=COMBINED_f)
        make_SC(COMBINED_f, NMT_SRC=NMT_SRC, AUG_SRC=AUG_SRC, TGT=NMT_TGT, COG_SRC=COG_SRC, COG_TGT=COG_TGT, sc_out_dir=aug_sc_dir)

def combine_csvs(csv1, csv2, out_csv):
    header1, data1 = read_csv(csv1)
    header2, data2 = read_csv(csv2)
    assert header1 == header2
    combined = [header1] + data1 + data2
    with open(out_csv, "w", newline='') as outf:
        writer = csv.writer(outf)
        for r in combined:
            writer.writerow(r)

def read_csv(f):
    with open(f, newline='') as inf:
        rows = [r for r in csv.reader(inf)]
    header = rows[0]
    data = rows[1:]
    return header, data

def read_config(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    NMT_SRC=None
    NMT_TGT=None
    AUG_SRC=None
    AUG_TGT=None
    SRC=None
    TGT=None
    for line in lines:
        if line.startswith("NMT_SRC="):
            assert NMT_SRC is None
            NMT_SRC = line.split("NMT_SRC=")[-1]
        elif line.startswith("NMT_TGT="):
            assert NMT_TGT is None
            NMT_TGT = line.split("NMT_TGT=")[-1]
        elif line.startswith("AUG_SRC="):
            assert AUG_SRC is None
            AUG_SRC = line.split("AUG_SRC=")[-1]
        elif line.startswith("AUG_TGT="):
            assert AUG_TGT is None
            AUG_TGT = line.split("AUG_TGT=")[-1]
        elif line.startswith("SRC="):
            assert SRC is None
            SRC = line.split("SRC=")[-1]
        elif line.startswith("TGT="):
            assert TGT is None
            TGT = line.split("TGT=")[-1]
    assert all([NMT_SRC is not None, NMT_TGT is not None, AUG_SRC is not None, AUG_TGT is not None, SRC is not None, TGT is not None])
    return NMT_SRC, NMT_TGT, AUG_SRC, AUG_TGT, SRC, TGT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SC_cfgs", "-c", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC", help="directory of cfgs for SC models")
    parser.add_argument("--NMT_DATA_DIR", "-d", default="/home/hatch5o6/Cognate/code/NMT/data")
    parser.add_argument("--aug_out", "-a", default="/home/hatch5o6/Cognate/code/NMT/augmented_data")
    parser.add_argument("--INSERT_NO_OVERLAP", default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    make_data(
        cfgs=args.SC_cfgs,
        nmt_data_dir=args.NMT_DATA_DIR,
        aug_out_dir=args.aug_out,
        INSERT_NO_OVERLAP=args.INSERT_NO_OVERLAP
    )
