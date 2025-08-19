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
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_/%j_%x.out
#SBATCH --job-name=hyper_param_search.fr-mfe.6.cfg.PREDICTION
#SBATCH --qos cs
#SBATCH --partition cs

nvidia-smi

bash Pipeline/pred_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH/fr-mfe.6.cfg
python Pipeline/clean_slurm_outputs.py
rm /home/hatch5o6/Cognate/code/core*
