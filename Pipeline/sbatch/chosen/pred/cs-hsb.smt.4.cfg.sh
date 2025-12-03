#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt/%j_%x.out
#SBATCH --job-name=SC_smt.pred.chosen.cs-hsb.smt.4.cfg
#SBATCH --qos dw87

bash Pipeline/pred_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/chosen/cs-hsb.smt.4.cfg
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*

