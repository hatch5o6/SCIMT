# TODO
import os
import argparse

def main(
    folder1,
    folder2
):
    SKIP_EXT = {"bin", "idx", "pt"}

    # folder1 = "/home/hatch5o6/nobackup/archive/CopperMT/bren_dan_compare2SAVE"
    # folder2 = "/home/hatch5o6/nobackup/archive/CopperMT/bren_danSAVE"
    f1name = folder1.split("/")[-1]
    f2name = folder2.split("/")[-1]

    print("F1 NAME:", f1name)
    print("F2 NAME:", f2name)

    equal = 0
    not_equal = 0
    for root, dirs, files in os.walk(folder1):
        for file in files:
            if any([file.endswith(ext) for ext in SKIP_EXT]):
                continue
            f1_path = os.path.join(root, file)
            # f2_path = f1_path.replace("bren_dan_compare2SAVE", "bren_danSAVE")
            f2_path = f1_path.replace(f"CopperMT/{f1name}", f"CopperMT/{f2name}")
            
            print("COMPARING")
            print("\t", f1_path)
            print("\t", f2_path)

            with open(f1_path) as inf:
                f1_content = inf.read()
            with open(f2_path) as inf:
                f2_content = inf.read()
            if f1_content == f2_content:
                equal += 1
                print("\tequal")
                print("------------------\n")
            else:
                not_equal += 1
                print("\tnot equal :(")
                print("------------------\n")

    print("EQUAL:", equal)
    print("NOT_EQUAL:", not_equal)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--folder1")
    parser.add_argument("-f2", "--folder2")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        folder1=args.folder1,
        folder2=args.folder2
    )
