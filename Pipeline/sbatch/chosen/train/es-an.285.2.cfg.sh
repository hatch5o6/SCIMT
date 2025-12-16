#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC/%j_%x.out
#SBATCH --job-name=SC.chosen.es-an.285.2.cfg
#SBATCH --qos cs
#SBATCH --partition cs

nvidia-smi

bash Pipeline/train_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/chosen/es-an.285.2.cfg
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*
