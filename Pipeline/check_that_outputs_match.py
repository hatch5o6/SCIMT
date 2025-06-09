# TODO
import os

SKIP_EXT = {"bin", "idx", "pt"}

folder1 = "/home/hatch5o6/nobackup/archive/CopperMT/bren_dan_compare2SAVE"
folder2 = "/home/hatch5o6/nobackup/archive/CopperMT/bren_danSAVE"

equal = 0
not_equal = 0
for root, dirs, files in os.walk(folder1):
    for file in files:
        if any([file.endswith(ext) for ext in SKIP_EXT]):
            continue
        f1_path = os.path.join(root, file)
        f2_path = f1_path.replace("bren_dan_compare2SAVE", "bren_danSAVE")
        
        with open(f1_path) as inf:
            f1_content = inf.read()
        with open(f2_path) as inf:
            f2_content = inf.read()
        if f1_content == f2_content:
            equal += 1
            # print("\tequal")
        else:
            print("COMPARING")
            print("\t", f1_path)
            print("\t", f2_path)
            not_equal += 1
            print("\tnot equal :(")
            print("------------------\n")

print("EQUAL:", equal)
print("NOT_EQUAL:", not_equal)
