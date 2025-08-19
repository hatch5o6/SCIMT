import argparse
import os
import shutil
import json

def copy_params(
    rnn_hyperparam_id,
    rnn_hyperparams_dir,
    copy_to_path
):
    manifest_f = os.path.join(rnn_hyperparams_dir, "manifest.json")
    with open(manifest_f) as inf:
        manifest = json.load(inf)
    rnn_hyperparams_f = manifest[rnn_hyperparam_id]
    rnn_hyperparams_f = os.path.join(rnn_hyperparams_dir, rnn_hyperparams_f)
    
    print(f"Copying {rnn_hyperparams_f} to {copy_to_path}")
    shutil.copyfile(rnn_hyperparams_f, copy_to_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--rnn_hyperparam_id")
    parser.add_argument("-d", "--rnn_hyperparams_dir")
    parser.add_argument("-c", "--copy_to_path")
    args = parser.parse_args()
    print("Arguments:-")
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}")
    print("------------------------------\n")
    return args

if __name__ == "__main__":
    print("###########################")
    print("# copy_rnn_hyperparams.py #")
    print("###########################")
    args = get_args()
    copy_params(
        rnn_hyperparam_id=args.rnn_hyperparam_id,
        rnn_hyperparams_dir=args.rnn_hyperparams_dir,
        copy_to_path=args.copy_to_path
    )