import os

data_dir = "/home/hatch5o6/Cognate/code/NMT/data"
standard_data = "/home/hatch5o6/Cognate/code/NMT/data/an-en_dev_test"
standard_test_f = os.path.join(standard_data, "test.csv")
standard_val_f = os.path.join(standard_data, "val.csv")

def read_file(f):
    with open(f) as inf:
        data = inf.read()
    return data

standard_test = read_file(standard_test_f)
standard_val = read_file(standard_val_f)

for d in os.listdir(data_dir):
    d = os.path.join(data_dir, d)
    val_f = os.path.join(d, "val.csv")
    if os.path.exists(val_f):
        print("checking", val_f)
        val = read_file(val_f)
        assert val == standard_val
    
    test_f = os.path.join(d, "test.csv")
    if os.path.exists(test_f):
        print("checking", test_f)
        test = read_file(test_f)
        assert test == standard_test

print("ALL PASSED")