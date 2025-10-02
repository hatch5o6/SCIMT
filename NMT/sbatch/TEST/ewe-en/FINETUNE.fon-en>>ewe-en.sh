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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/ewe-en/%j_%x.out
#SBATCH --job-name=TEST.ewe-en.FINETUNE.fon-en>>ewe-en
#SBATCH --qos=dw87

python NMT/clean_slurm_outputs.py

nvidia-smi
srun python NMT/train.py \
	--config /home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/ewe-en/FINETUNE.fon-en>>ewe-en.yaml \
	--mode TEST


python NMT/clean_slurm_outputs.py
