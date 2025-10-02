import os
import shutil
import argparse
from tqdm import tqdm
import yaml

CONFIGS_DIR="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS"
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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/{LANG_OUT}/%j_%x.out
#SBATCH --job-name={name}
#SBATCH --qos={qos}

python NMT/clean_slurm_outputs.py

nvidia-smi
srun {python_command}

python NMT/clean_slurm_outputs.py
""".strip()

def main(configs_dir, mode, qos, out_dir):
    out_dir = os.path.join(out_dir, mode)
    if os.path.exists(out_dir):
        print("Deleting:", out_dir)
        shutil.rmtree(out_dir)
    print("Creating:", out_dir)
    os.mkdir(out_dir)

    for d in tqdm(os.listdir(configs_dir)):
        if d in ["data_log.csv", "data_params_log.csv"]: continue
        # print("D:", d)
        d_config = os.path.join(configs_dir, d)
        d_out = os.path.join(out_dir, d)
        assert not os.path.exists(d_out)
        os.mkdir(d_out)

        sbatch_fs = []
        for f in os.listdir(d_config):
            # print(f"F:", f)
            assert f.endswith(".yaml")
            f_config = os.path.join(d_config, f)
            config = read_config(f_config)
            f_out = os.path.join(d_out, f)[:-4] + "sh"
            assert not os.path.exists(f_out)
            
            python_command = f"python NMT/train.py \\\n\t--config {f_config} \\\n\t--mode {mode}\n"
            name=f"{mode}.{d}.{f[:-5]}"

            if mode == "TRAIN":
                n_gpus = config["n_gpus"]
            else:
                n_gpus = 1

            sbatch_content = sbatch_template.replace("{name}", name) \
                .replace("{qos}", qos) \
                .replace("{python_command}", python_command) \
                .replace("{LANG_OUT}", d) \
                .replace("{n_gpus}", str(n_gpus))
            
            lang_out_dir = f"/home/hatch5o6/Cognate/code/NMT/slurm_outputs/{d}"
            if not os.path.exists(lang_out_dir):
                os.mkdir(lang_out_dir)

            with open(f_out, "w") as outf:
                outf.write(sbatch_content + "\n")
            sbatch_fs.append(f_out)

        start_all_f = os.path.join(d_out, "all_except_finetune.sh")
        with open(start_all_f, "w") as outf:
            for sf in sbatch_fs:
                sf_name = sf.split("/")[-1]
                if not (sf_name.startswith("FINETUNE.") or sf_name.startswith("CHAR-FINETUNE.")):
                    outf.write(f"sbatch {sf}\n")

def read_config(f):
    with open(f) as inf:
        config = yaml.safe_load(inf)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs_dir", default="/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS")
    parser.add_argument("--out", default="/home/hatch5o6/Cognate/code/NMT/sbatch")
    parser.add_argument("--qos", default="dw87", choices=["dw87"])
    parser.add_argument("--mode", choices=["TRAIN", "TEST", "INFERENCE"])
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(
        configs_dir=args.configs_dir,
        mode=args.mode,
        qos=args.qos,
        out_dir=args.out
    )