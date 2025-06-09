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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%x.out
#SBATCH --job-name=hr_data_preprocess.es_an.es2en.10M.train.PART3.sh
#SBATCH --qos=dw87

sh sh/hr_data_preprocess.es_an.es2en.10M.train.PART3.sh
