#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/slurm_outputs/%j_%x.out
#SBATCH --job-name=train_chosen_tokenizer.es-an
#SBATCH --qos dw87

bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/es-an_en.cfg
bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/es2an-an_en.0.cfg
bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/es2an-an_en.2.cfg