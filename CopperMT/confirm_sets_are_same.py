import os

dir_1 = "/home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/inputs/split_data/cs_hsb/0"
dir_2 = "/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/cs_hsb/0"

for f in os.listdir(dir_1):
    data1 = None
    data2 = None
    f_v1 = os.path.join(dir_1, f)
    f_v2 = os.path.join(dir_2, f)
    assert f_v1 != f_v2
    with open(f_v1) as inf1, open(f_v2) as inf2:
        data1 = inf1.read()
        data2 = inf2.read()
    assert isinstance(data1, str)
    assert isinstance(data2, str)
    print("------------------")
    print("F1", f_v1)
    print(f"\t`{data1[:15]}`...\n\t...`{data1[-15:]}`")
    print("F2", f_v2)
    print(f"\t`{data2[:15]}`...\n\t...`{data2[-15:]}`")
    assert data1 == data2

print("PASSED ALL TESTS")
