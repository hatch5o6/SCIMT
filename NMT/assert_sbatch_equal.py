import os

TEST_DIR="/home/hatch5o6/Cognate/code/NMT/sbatch/TEST"
GT_TEST_DIR="/home/hatch5o6/Cognate/code/NMT/sbatch/TEST_GT"

TRAIN_DIR="/home/hatch5o6/Cognate/code/NMT/sbatch/TRAIN"
GT_TRAIN_DIR="/home/hatch5o6/Cognate/code/NMT/sbatch/TRAIN_GT"


def test_folder(folder, gt_folder):
    folder_ds = os.listdir(folder)
    gt_folder_ds = os.listdir(gt_folder)

    assert sorted(folder_ds) == sorted(gt_folder_ds)
    TOTAL_DIRS = 0
    DIRS_PASSED = 0
    for i, d in enumerate(gt_folder_ds):
        TOTAL_DIRS += 1
        gt_d_path = os.path.join(gt_folder, d)
        d_path = os.path.join(folder, d)
        print(f"-------------------------------- ({i}) ------------------------------------")
        print("COMPARING GT:", gt_d_path)
        print("          TO:", d_path)
        dir_passed = compare_dirs(gt_d_path, d_path)
        if dir_passed:
            DIRS_PASSED += 1
            print("DIR PASSED")
        else:
            print("DIR FAILED")
    print(f"{DIRS_PASSED} / {TOTAL_DIRS} DIRS PASSED")

def compare_dirs(gt_d, d):
    assert os.listdir(gt_d) == os.listdir(d)
    TOTAL = 0
    PASSED = 0
    for f in os.listdir(gt_d):
        TOTAL += 1
        gt_f = os.path.join(gt_d, f)
        f = os.path.join(d, f)
        print("\t------")
        print("\tcomparing gt:", gt_f)
        print("\t          to:", f)
        if read_f(gt_f) == read_f(f):
            print("\t          passed:)")
            PASSED += 1
    print(f"\tPASSED {PASSED} / {TOTAL}")
    if PASSED == TOTAL:
        print("\t\tALL FILES PASSED")
    return TOTAL == PASSED

            
def read_f(f):
    with open(f) as inf:
        content = inf.read()
    return content
    

print("TRAIN TESTS:")
test_folder(TRAIN_DIR, GT_TRAIN_DIR)


print("\n\n\n\n\nTEST TESTS:")
test_folder(TEST_DIR, GT_TEST_DIR)