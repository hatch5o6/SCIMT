import os

SBATCH_TEMPLATE="""
#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt/%j_%x.out
#SBATCH --job-name=SC_smt.{NAME}
#SBATCH --qos dw87

bash Pipeline/train_SC.sh {CFG}
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*
""".lstrip()

CFG_DIR="/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT"
SBATCH_OUT_DIR="/home/hatch5o6/Cognate/code/Pipeline/sbatch/smt"


for f in os.listdir(CFG_DIR):
    assert f.endswith(".smt.cfg")
    f_path = os.path.join(CFG_DIR, f)

    lang_pair = f[:-8]

    sbatch_f = os.path.join(SBATCH_OUT_DIR, f"{lang_pair}.smt.cfg.sh")
    sbatch_content = SBATCH_TEMPLATE.replace("{NAME}", lang_pair).replace("{CFG}", f_path)
    with open(sbatch_f, "w") as outf:
        outf.write(sbatch_content + "\n")

