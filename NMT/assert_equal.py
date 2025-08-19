import os

dirs = [
    # "/home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN",
    # "/home/hatch5o6/Cognate/code/NMT/augmented_data/SC",
    "/home/hatch5o6/Cognate/code/NMT/data/SC"
]
files = []
for d in dirs:
    dir_files = [os.path.join(d, f) for f in os.listdir(d)]
    files += dir_files

total = 0 
passed = 0
for f in files:
    f_name = f.split("/")[-1]
    # print(f_name)
    if f_name.endswith("_dev_test"):
        continue
    file_dir = "/".join(f.split("/")[:-1])
    parent_dir = "/".join(file_dir.split("/")[:-1])
    file_dir_name = file_dir.split("/")[-1]
    compare_file_dir_name = "_" + file_dir_name
    compare_file_dir = os.path.join(parent_dir, compare_file_dir_name)
    compare_file = os.path.join(compare_file_dir, f_name)

    print("---------------------------")
    print("f         :", f)
    print("compare to:", compare_file)
    for fi in os.listdir(f):
        total += 1
        print("fi", fi)
        fipath = os.path.join(f, fi)
        compareto = compare_file + "/" + fi
        
        print("\tcomparing", fipath)
        print("\t\tto", compareto)
        assert fipath != compareto

        with open(fipath) as inf:
            fi_content = inf.read()
            print("FI: ", fi_content[:200])
        with open(compareto) as inf:
            compareto_content = inf.read()
            print("COMP: ", compareto_content[:200])
        assert fi_content == compareto_content
        if fi_content == compareto_content:
            passed += 1

print("TOTAL", total)
print("PASSED", passed)
print("ALL PASSED")





