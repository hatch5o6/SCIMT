#!/bin/bash

#SBATCH --time=72:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=4
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/hsb-de/%j_%x.out
#SBATCH --job-name=TEST.hsb-de.PRETRAIN.SC_cs2hsb-de
#SBATCH --qos=dw87

nvidia-smi
python train.py \
	--config /home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/hsb-de/PRETRAIN.SC_cs2hsb-de.yaml \
	--mode TEST


python clean_slurm_outputs.py
