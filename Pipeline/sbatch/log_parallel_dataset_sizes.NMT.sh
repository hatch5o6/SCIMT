#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/%j_%x.out
#SBATCH --job-name=log_parallel_dataset_sizes.NMT.sh
#SBATCH --qos=matrix

python Pipeline/clean_slurm_outputs.py

python Pipeline/log_parallel_dataset_sizes.py \
    --nmt_data_dir /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN \
    --LOG_F /home/hatch5o6/Cognate/code/NMT_parallel_dataset_log.json

python Pipeline/clean_slurm_outputs.py