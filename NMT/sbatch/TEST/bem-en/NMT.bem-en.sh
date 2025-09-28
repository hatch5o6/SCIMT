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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/bem-en/%j_%x.out
#SBATCH --job-name=TEST.bem-en.NMT.bem-en
#SBATCH --qos=dw87

nvidia-smi
python train.py \
	--config /home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/bem-en/NMT.bem-en.yaml \
	--mode TEST


python clean_slurm_outputs.py
