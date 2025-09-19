import yaml
import os


def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def assert_train(file):
    assert isinstance(file, str)
    assert file != ""
    assert "," not in file
    assert file.endswith("/train.no_overlap_v1.csv")

def assert_val(file):
    assert isinstance(file, str)
    assert file != ""
    assert "," not in file
    assert file.endswith("/val.no_overlap_v1.csv")

def assert_test(file):
    assert isinstance(file, str)
    assert file != ""
    assert "," not in file
    assert file.endswith("/test.csv")

CONFIGS_DIR = "/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS"

for d in os.listdir(CONFIGS_DIR):
    d_path = os.path.join(CONFIGS_DIR, d)
    for f in os.listdir(d_path):
        f_path = os.path.join(d_path, f)
        print("f_path:", f_path)
        config = read_config(f_path)
        assert_train(config["train_data"])
        assert_val(config["val_data"])
        assert_test(config["test_data"])
        print("\tpassed :)")
print("ALL PASSED")
        
