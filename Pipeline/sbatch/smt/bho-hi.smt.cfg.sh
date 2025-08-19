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
#SBATCH --job-name=SC_smt.bho-hi
#SBATCH --qos dw87

bash Pipeline/train_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC_SMT/bho-hi.smt.cfg
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*

