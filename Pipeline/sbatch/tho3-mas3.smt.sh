#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/test_slurm_outputs/%j_%x.out
#SBATCH --job-name=tho3-mas3.SC_pipeline

bash Pipeline/train_SC.sh Pipeline/cfg/SC/tho3-mas3.cfg
python Pipeline/clean_slurm_outputs.py