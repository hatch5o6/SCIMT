import argparse
import os
import shutil
from tqdm import tqdm

def main(
    SC_dir,
    out_dir
):
    if os.path.exists(out_dir):
        print("Removing", out_dir)
        shutil.rmtree(out_dir)
    print("Creating", out_dir)
    os.mkdir(out_dir)

    for f in tqdm(os.listdir(SC_dir)):
        assert f.endswith(".cfg")
        if f.endswith(".smt.cfg"):
            continue

        f_path = os.path.join(SC_dir, f)
        smt_f_path = os.path.join(out_dir, f[:-3] + "smt.cfg")
        assert smt_f_path != f_path
        print("-----------------------------------")
        print("reading", f_path)
        print("writing", smt_f_path)
        with open(f_path) as inf, open(smt_f_path, "w") as outf:
            f_lines = [l.rstrip() for l in inf.readlines()]
            for f_line in f_lines:
                if f_line == "SC_MODEL_TYPE=RNN":
                    outf.write("SC_MODEL_TYPE=SMT\n")
                elif f_line.startswith("RNN_HYPERPARAMS="):
                    outf.write("RNN_HYPERPARAMS=null\n")
                elif f_line.startswith("RNN_HYPERPARAMS_ID="):
                    outf.write("RNN_HYPERPARAMS_ID=null\n")
                elif f_line.startswith("SC_MODEL_ID="):
                    key, sc_model_id = tuple(f_line.split("="))
                    assert key == "SC_MODEL_ID"
                    assert "-RNN-" in sc_model_id
                    smt_sc_model_id = sc_model_id.replace("-RNN-", "-SMT-")
                    outf.write(f"{key}={smt_sc_model_id}\n")
                else:
                    outf.write(f_line + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SC_dir", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC")
    parser.add_argument("--out_dir", default="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        SC_dir=args.SC_dir,
        out_dir=args.out_dir
    )