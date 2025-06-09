import argparse
import os

COPPERMT_DIR="/home/hatch5o6/Cognate/code/CopperMT/CopperMT"
parameters_stensil = """
MEDeA_DIR="/home/hatch5o6/nobackup/archive/CopperMT/{SRC}_{TGT}"

WK_DIR="${MEDeA_DIR}/workspace"
INPUTS_DIR="${MEDeA_DIR}/inputs"
lang="{SRC}-{TGT}"

DATA_NAME="{SRC}_{TGT}"

MOSES_DIR="/home/hatch5o6/Cognate/code/CopperMT/CopperMT/submodules"
""".strip()

sbatch_preamble = """
#!/bin/bash

#SBATCH --time={HOURS}:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus={N_GPUS}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/%x.out
#SBATCH --job-name={JOB_NAME}
""".strip()

# SBATCH_DIR = "/home/hatch5o6/Cognate/code/Pipeline/sbatch"
# Will somehow need to select the best model and choose it for prediction.

def write(
    src,
    tgt,
    parameters_f
    # train_sc_f,
    # predict_sc_f,
    # TYPE
):
    write_parameters(parameters_f, src, tgt)

def write_parameters(f, src, tgt):
    assert f.endswith(".cfg")
    ending = f"{src}-{tgt}.cfg"
    if not f.endswith(ending):
        f = f[:-3] + f"{src}-{tgt}.cfg"

    content = parameters_stensil
    content = content.replace("{SRC}", src)
    content = content.replace("{TGT}", tgt)
    with open(f, "w") as outf:
        outf.write(content + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("-p", "--parameters", default="parameters.cfg", required=True)
    # parser.add_argument("-t", "--train_sc", required=True)
    # parser.add_argument("-h", "--predict_sc", required=True)
    # parser.add_argument("--TYPE", choices=["RNN", "SMT"])
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("####################")
    print("# write_scripts.py #")
    print("####################")
    args = get_args()
    assert args.parameters.endswith(".cfg")
    write(
        src=args.src,
        tgt=args.tgt,
        parameters_f = args.parameters
        # train_sc_f = args.train_sc,
        # predict_sc_f = args.predict_sc,
        # TYPE=args.TYPE
    )
