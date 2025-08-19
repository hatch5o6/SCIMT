import argparse
import os

COPPERMT_DIR="/home/hatch5o6/Cognate/code/CopperMT/CopperMT"
parameters_stensil = """
MEDeA_DIR="{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}"

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
    coppermt_data_dir,
    sc_model_type,
    rnn_hyperparams_id,
    seed,
    parameters_f
):
    write_parameters(coppermt_data_dir, parameters_f, src, tgt, sc_model_type, rnn_hyperparams_id, seed)

def write_parameters(coppermt_data_dir, f, src, tgt, sc_model_type, rnn_hyperparams_id, seed):
    assert f.endswith(".cfg")
    # ending = f"{src}-{tgt}.cfg"
    # if not f.endswith(ending):
    #     f = f[:-3] + f"{src}-{tgt}.cfg"

    content = parameters_stensil
    content = content.replace("{COPPERMT_DATA_DIR}", coppermt_data_dir)
    content = content.replace("{SRC}", src)
    content = content.replace("{TGT}", tgt)
    content = content.replace("{SC_MODEL_TYPE}", sc_model_type)
    content = content.replace("{RNN_HYPERPARAMS_ID}", rnn_hyperparams_id)
    content = content.replace("{SEED}", seed)
    with open(f, "w") as outf:
        outf.write(content + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--tgt", required=True)
    parser.add_argument("--coppermt_data_dir", required=True)
    parser.add_argument("--sc_model_type", required=True)
    parser.add_argument("--rnn_hyperparams_id", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("-p", "--parameters", default="parameters.cfg", required=True)
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
        coppermt_data_dir=args.coppermt_data_dir,
        sc_model_type=args.sc_model_type,
        rnn_hyperparams_id=args.rnn_hyperparams_id,
        seed=args.seed,
        parameters_f = args.parameters
    )
