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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/fon-fr/%j_%x.out
#SBATCH --job-name=TEST.fon-fr.NMT.fon-fr
#SBATCH --qos=dw87

nvidia-smi
python train.py \
	--config /home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/fon-fr/NMT.fon-fr.yaml \
	--mode TEST


python clean_slurm_outputs.py
