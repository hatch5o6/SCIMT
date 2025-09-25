#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/predict/%j_%x.out
#SBATCH --job-name=predict.cs-hsb.smt.cfg
#SBATCH --qos dw87

nvidia-smi

bash Pipeline/pred_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT/cs-hsb.smt.cfg
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*
