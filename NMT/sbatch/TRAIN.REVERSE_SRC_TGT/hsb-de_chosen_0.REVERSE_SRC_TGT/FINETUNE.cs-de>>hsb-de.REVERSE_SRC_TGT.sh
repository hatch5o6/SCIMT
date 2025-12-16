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
#SBATCH --output /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/hsb-de_chosen_0/%j_%x.out
#SBATCH --job-name=TRAIN.hsb-de_chosen_0.FINETUNE.cs-de>>hsb-de
#SBATCH --qos=dw87



python NMT/clean_slurm_outputs.py

nvidia-smi

# tensorboard --logdir "{tb_dir}" --port 6006 --host 0.0.0.0 &
srun python NMT/train.py \
	--config "/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/hsb-de_chosen_0/FINETUNE.cs-de>>hsb-de.yaml" \
	--mode TRAIN \
	--REVERSE_SRC_TGT


python NMT/clean_slurm_outputs.py
