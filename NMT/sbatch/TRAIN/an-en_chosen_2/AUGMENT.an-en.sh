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
#SBATCH --output /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en_chosen_2/%j_%x.out
#SBATCH --job-name=TRAIN.an-en_chosen_2.AUGMENT.an-en
#SBATCH --qos=dw87



python NMT/clean_slurm_outputs.py

nvidia-smi

# tensorboard --logdir "{tb_dir}" --port 6006 --host 0.0.0.0 &
srun python NMT/train.py \
	--config "/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/an-en_chosen_2/AUGMENT.an-en.yaml" \
	--mode TRAIN


python NMT/clean_slurm_outputs.py
