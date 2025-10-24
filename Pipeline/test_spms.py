import argparse
import os

TEST_FOLDER="/home/hatch5o6/nobackup/archive/CognateMT/spm_models_TEST"
"""
This is for testing train_srctgt_tokenizer.sh / train_all_tokenizers.sh
This ensures that the spm data and models created match those of the TEST_FOLDER.
"""

def main(spm_dir):
    TOTAL = 0
    PASSED = 0
    for d in os.listdir(TEST_FOLDER):
        if d not in os.listdir(spm_dir):
            print(f"{d} in {TEST_FOLDER} but not in {spm_dir}")
        assert d in os.listdir(spm_dir)
        if d == "notes":
            continue
        
        d_gt_path = os.path.join(TEST_FOLDER, d)
        d_hyp_path = os.path.join(spm_dir, d)

        print("####################################################################################")
        print(" GT:", d_gt_path)
        print("HYP:", d_hyp_path)

        gt_files = []
        hyp_files = []

        for f in os.listdir(d_gt_path):
            if f == "notes":
                print(f"DOES {f} EXIST IN HYP DIR?")
                assert f in os.listdir(d_hyp_path)
                print("\t YES :)")
                continue
            f_gt_path = os.path.join(d_gt_path, f)
            f_hyp_path = os.path.join(d_hyp_path, f)

            if os.path.isdir(f_gt_path):
                for sub_f in os.listdir(f_gt_path):
                    if sub_f.endswith(".model"):
                        print(f"DOES {sub_f} EXIST IN HYP DIR?")
                        assert sub_f in os.listdir(f_hyp_path)
                        print("\t YES :)")
                        continue
                    sub_f_gt_path = os.path.join(f_gt_path, sub_f)
                    sub_f_hyp_path = os.path.join(f_hyp_path, sub_f)
                    gt_files.append(sub_f_gt_path)
                    hyp_files.append(sub_f_hyp_path)
            else:
                gt_files.append(f_gt_path)
                hyp_files.append(f_hyp_path)
        
        assert len(gt_files) == len(hyp_files)
        pairs = list(zip(gt_files, hyp_files))

        for gt_f, hyp_f in pairs:
            TOTAL += 1
            print("-------------------------------------")
            print("COMPARING")
            print(" GT F:", gt_f)
            print("HYP F:", hyp_f)
            if read_f(gt_f) == read_f(hyp_f):
                print("\tpassed :)")
                PASSED += 1
            else:
                print("\tfailed!!!")
    
    print(f"\n\nPASSED {PASSED} / {TOTAL}")

def read_f(f):
    print("reading:", f)
    with open(f) as inf:
        content = inf.read()
    return content

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spm_dir", default="/home/hatch5o6/nobackup/archive/CognateMT/spm_models", help="the folder containing the spm data and models we are testing")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(
        args.spm_dir
    )