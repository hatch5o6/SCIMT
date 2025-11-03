#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%j_%x.out
#SBATCH --job-name=make_new_CharLOTTE_data.TEST.arabic.sh
#SBATCH --qos=cs
#SBATCH --partition=cs

python make_new_CharLOTTE_data.TEST.arabic.py