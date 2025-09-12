import os
import shutil
import traceback
import sys

SC_DIR = "/home/hatch5o6/Cognate/code/Pipeline/cfg/SC"
SC_SMT_DIR = "/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT_OLD"
NEW_SC_SMT_DIR = "/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT"

def main():
    if os.path.exists(NEW_SC_SMT_DIR):
        shutil.rmtree(NEW_SC_SMT_DIR)
    os.mkdir(NEW_SC_SMT_DIR)

    SC_cfgs = os.listdir(SC_DIR)
    SC_SMT_cfgs = os.listdir(SC_SMT_DIR)
    assert len(SC_cfgs) == len(SC_SMT_cfgs)

    for sc_cfg in SC_cfgs:
        pair = sc_cfg.split(".")[0]

        src, tgt = pair.split("-")
        assert src.strip() != ""
        assert tgt.strip() != ""

        sc_smt_cfg = f"{pair}.smt.cfg"
        assert sc_smt_cfg in SC_SMT_cfgs

        sc_cfg_path = os.path.join(SC_DIR, sc_cfg)
        sc_smt_cfg_path = os.path.join(SC_SMT_DIR, sc_smt_cfg)

        sc_config = read_cfg(sc_cfg_path)
        sc_smt_config = read_cfg(sc_smt_cfg_path)
        assert set(sc_config.keys()) == set(sc_smt_config.keys())

        try:
            for key in sc_config.keys():
                sc_value = sc_config[key]
                sc_smt_value = sc_smt_config[key]

                if key in ["PARALLEL_TRAIN", "PARALLEL_VAL"]:
                    assert sc_value.endswith(".no_overlap_v1.csv")
                    assert sc_smt_value == sc_value[:-18] + ".csv"
                    sc_smt_config[key] = sc_value
                elif key == "APPLY_TO":
                    assert (sc_value == sc_smt_value == "null") or (sc_value != sc_smt_value)
                    if sc_value != sc_smt_value:
                        sc_file_list = sc_value.split(",")
                        sc_smt_file_list = sc_smt_value.split(",")
                        assert len(sc_file_list) == len(sc_smt_file_list)
                        for scf, sc_smtf in list(zip(sc_file_list, sc_smt_file_list)):
                            if scf.endswith("/train.no_overlap_v1.csv") or scf.endswith("/val.no_overlap_v1.csv"):
                                assert sc_smtf == scf[:-18] + ".csv"
                            elif scf.endswith("/test.csv"):
                                assert sc_smtf == scf
                            else:
                                assert False
                        sc_smt_config[key] = sc_value
                elif key == "SC_MODEL_TYPE":
                    assert sc_value == "RNN"
                    assert sc_smt_value == "SMT"
                elif key == "RNN_HYPERPARAMS":
                    assert sc_value == "/home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams"
                    assert sc_smt_value == "null"
                elif key == "RNN_HYPERPARAMS_ID":
                    assert sc_value == "0"
                    assert sc_smt_value == "null"
                elif key == "SC_MODEL_ID":
                    assert sc_value == f"{src.upper()}-{tgt.upper()}-RNN-0"
                    assert sc_smt_value == f"{src.upper()}-{tgt.upper()}-SMT-0"
                else:
                    assert sc_value == sc_smt_value
        except Exception as e:
            # Get the traceback information
            exc_type, exc_obj, exc_tb = sys.exc_info()
            # Extract the line number from the traceback object
            line_number = exc_tb.tb_lineno
            print(f"Exception occurred on line: {line_number}")
            # You can also print the full traceback for more detailed information
            traceback.print_exc()

            print("KEY:", key)
            print("sc_cfg:", sc_cfg)
            print("\tvalue:", sc_value)
            print("sc_smt_cfg:", sc_smt_cfg)
            print("\tvalue:", sc_smt_value)
            exit()
    
        new_sc_smt_content = []
        with open(sc_smt_cfg_path) as inf:
            lines = [l.strip() for l in inf.readlines()]
            for line in lines:
                if line.startswith("#") or line.strip() == "":
                    new_sc_smt_content.append(line + "\n")
                    continue

                # print(f"NEW SC SMT LINE: `{line}`")
                # split_line = line.split("=")
                split_line = split_line_on_first_equals(line)
                assert len(split_line) == 2
                key, value = tuple(split_line)
                assert key.strip() != ""
                assert value.strip() != ""
                assert key in sc_smt_config
                # print("---------------")
                # print("pair", pair)
                # print("KEY", key)
                # print("value", value)
                # print("sc_smt_config[key]", sc_smt_config[key])
                if key in ["PARALLEL_TRAIN", "PARALLEL_VAL"]:
                    assert value != sc_smt_config[key]
                    new_sc_smt_content.append(f"{key}={sc_smt_config[key]}\n")
                elif key == "APPLY_TO":
                    if not (value == sc_smt_config[key] == "null"):
                        assert value != sc_smt_config[key]
                    new_sc_smt_content.append(f"{key}={sc_smt_config[key]}\n")
                else:
                    assert value == sc_smt_config[key]
                    new_sc_smt_content.append(line + "\n")
        
        new_sc_smt_path = os.path.join(NEW_SC_SMT_DIR, sc_smt_cfg)
        assert not os.path.exists(new_sc_smt_path)
        with open(new_sc_smt_path, "w") as outf:
            outf.write("".join(new_sc_smt_content))

        new_sc_smt_config = read_cfg(new_sc_smt_path)
        # print(sorted(list(sc_config.keys())))
        # print("\n")
        # print(sorted(list(new_sc_smt_config.keys())))
        assert set(sc_config.keys()) == set(new_sc_smt_config.keys())
        for key in sc_config.keys():
            sc_value = sc_config[key]
            new_sc_smt_value = new_sc_smt_config[key]
            if key == "SC_MODEL_TYPE":
                assert sc_value == "RNN"
                assert new_sc_smt_value == "SMT"
            elif key == "RNN_HYPERPARAMS":
                assert sc_value == "/home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams"
                assert new_sc_smt_value == "null"
            elif key == "RNN_HYPERPARAMS_ID":
                assert sc_value == "0"
                assert new_sc_smt_value == "null"
            elif key == "SC_MODEL_ID":
                assert sc_value == f"{src.upper()}-{tgt.upper()}-RNN-0"
                assert new_sc_smt_value == f"{src.upper()}-{tgt.upper()}-SMT-0"
            else:
                assert sc_value == new_sc_smt_value
                if key in ["PARALLEL_TRAIN", "PARALLEL_VAL"]:
                    assert sc_value.endswith(".no_overlap_v1.csv")
                    assert new_sc_smt_value.endswith(".no_overlap_v1.csv")
                elif key == "APPLY_TO":
                    if not (sc_value == new_sc_smt_value == "null"):
                        sc_list = sc_value.split(",")
                        new_sc_smt_list = new_sc_smt_value.split(",")
                        assert sc_list == new_sc_smt_list
                        # print("PAIR:", pair)
                        # print("sc_value", sc_value)
                        # print("new_sc_smt_value", new_sc_smt_value)
                        for f in sc_list:
                            assert any([
                                f.endswith("/train.no_overlap_v1.csv"),
                                f.endswith("/val.no_overlap_v1.csv"),
                                f.endswith("/test.csv")
                            ])
        print(f"{pair} passed")                    
    print("ALL NEW SC SMT FILES PASSED TEST :)")

def split_line_on_first_equals(line):
    prefix = line.split("=")[0]
    the_rest = "=".join(line.split("=")[1:])
    return [prefix, the_rest]

def read_cfg(f):
    print(f)
    config = {}
    with open(f) as inf:
        lines = [l.strip() for l in inf.readlines()]
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        else:
            # split_line = line.split("=")
            split_line = split_line_on_first_equals(line)
            if not (len(split_line) == 2):
                print(f"SPLIT_LINE:", split_line)
            assert len(split_line) == 2
            key, value = tuple(split_line)
            assert key.strip() != ""
            assert value.strip() != ""
            assert key not in config
            config[key] = value
    return config

if __name__ == "__main__":
    main()