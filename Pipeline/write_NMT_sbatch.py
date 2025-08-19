import argparse
import os
import shutil
import yaml
from tqdm import tqdm

F_TYPES = ["AUGMENT", "FINETUNE", "NMT", "PRETRAIN"]

SKIP = ["_OLD", "preliminary", "TEST", "template.yaml"]

sbatch_template = """
#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node={n_gpus}
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus={n_gpus}
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%j_%x.out
#SBATCH --job-name={job_name}
#SBATCH --qos={qos}
""".strip()

def write_NMT_sbatch_files(
    sbatch_folder,
    configs_folder
):
    if os.path.exists(sbatch_folder):
        print("removing", sbatch_folder)
        shutil.rmtree(sbatch_folder)
    print("creating", sbatch_folder)
    os.mkdir(sbatch_folder)

    for d in tqdm(os.listdir(configs_folder)):
        if d in SKIP: continue
        if "-" not in d: continue
        split_d = [x for x in d.split("-") if x.strip() != ""]
        if len(split_d) != 2: continue

        d_path = os.path.join(configs_folder, d)
        if not os.path.isdir(d_path): continue

        print("Lang pair dir:", d_path)
        for f in os.listdir(d_path):
            assert f.endswith(".yaml")
            f_path = os.path.join(d_path, f)
            config = read_yaml(f_path)

            f_type = f.split(".")[0]
            assert f_type in F_TYPES
            f_type_folder = os.path.join(sbatch_folder, f_type)
            f_type_train = os.path.join(f_type_folder, "train")
            f_type_test = os.path.join(f_type_folder, "test")
            if not os.path.exists(f_type_folder):
                os.mkdir(f_type_folder)
                os.mkdir(f_type_train)
                os.mkdir(f_type_test)

            sbatch_content = sbatch_template.replace("{n_gpus}", str(config["n_gpus"]))
            sbatch_content = sbatch_content.replace("{qos}", config["qos"])
            if config["qos"] == "cs":
                sbatch_content += "\n--partition=cs"
            sbatch_content = sbatch_content.replace("{job_name}", f.split(".yaml")[0])
            sbatch_content += "\n\nnvidia-smi\n"
            sbatch_content += f"python train.py \\\n\t--config {f} \\\n\t--mode TRAIN\n"
            sbatch_content += "\npython clean_slurm_outputs.py\n"

            sbatch_path = os.path.join(
                f_type_train,
                "train_" + f.split(".yaml")[0] + ".sh"
            )
            # print("writing to", sbatch_path)
            with open(sbatch_path, "w") as outf:
                outf.write(sbatch_content)
            
            sbatch_test_path = os.path.join(
                f_type_test,
                "test_" + f.split(".yaml")[0] + ".sh"
            )
            sbatch_test_content = sbatch_content.replace("--mode TRAIN", "--mode TEST")
            # print("writing to", sbatch_test_path)
            with open(sbatch_test_path, "w") as outf:
                outf.write(sbatch_test_content)

def read_yaml(f):
    with open(f, 'r') as inf:
        data = yaml.safe_load(inf)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sbatch_folder", default="/home/hatch5o6/Cognate/code/NMT/sbatch")
    parser.add_argument("-c", "--configs_folder", default="/home/hatch5o6/Cognate/code/NMT/configs")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    write_NMT_sbatch_files(
        sbatch_folder=args.sbatch_folder, 
        configs_folder=args.configs_folder
    )