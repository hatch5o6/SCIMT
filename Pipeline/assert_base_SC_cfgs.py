import argparse
import os
import json

def main(SC_dir):
    for f in os.listdir(SC_dir):
        f = os.path.join(SC_dir, f)

        print("\n------------------------------------------------")
        print("CHECKING", f)

        assert f.endswith(".cfg")
        settings = read_cfg(f)

        assert settings["MODULE_HOME_DIR"] == "/home/hatch5o6/Cognate/code"
        assert settings["SEED"] == "0"
        assert settings["NO_GROUPING"] == "true"

        if f.endswith(".smt.cfg"):
            assert settings["SC_MODEL_TYPE"] == "SMT"
        else:
            assert settings["SC_MODEL_TYPE"] == "RNN"
        
        assert settings["COGNATE_THRESH"] == "0.5"
        assert settings["COPPERMT_DATA_DIR"] == "/home/hatch5o6/nobackup/archive/CopperMT"
        assert settings["COPPERMT_DIR"] == "/home/hatch5o6/Cognate/code/CopperMT/CopperMT"
        assert settings["PARAMETERS_DIR"] == "/home/hatch5o6/Cognate/code/Pipeline/parameters"
        assert settings["RNN_HYPERPARAMS"] == "/home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams"

        if f.endswith(".smt.cfg"):
            assert settings["RNN_HYPERPARAMS_ID"] == "null"
        else:
            assert settings["RNN_HYPERPARAMS_ID"] == "0"

        assert settings["BEAM"] == "5"
        assert settings["NBEST"] == "1"
        assert settings["REVERSE_SRC_TGT_COGNATES"] == "false"

        for k in [
            "SC_MODEL_ID",
            "COGNATE_TRAIN_RATIO",
            "COGNATE_VAL_RATIO",
            "COGNATE_TEST_RATIO",
            "VAL_COGNATES_SRC",
            "VAL_COGNATES_TGT",
            "TEST_COGNATES_SRC",
            "TEST_COGNATES_TGT",
            "ADDITIONAL_TRAIN_COGNATES_SRC",
            "ADDITIONAL_TRAIN_COGNATES_TGT"
        ]:
            print(f"asserting {k} in settings")
            assert k in settings
    
    print("\n\nALL PASSED")


def read_cfg(f):
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    settings = {}
    for line in lines:
        if line == "":
            continue
        if line.startswith("#"):
            continue
        split_line = line.split("=")
        key = split_line[0]
        value = "=".join(split_line[1:])
        assert key not in settings
        settings[key] = value
    
    print("SETTINGS")
    print(json.dumps(settings, ensure_ascii=False, indent=2))
    return settings

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SC_dir", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(SC_dir=args.SC_dir)