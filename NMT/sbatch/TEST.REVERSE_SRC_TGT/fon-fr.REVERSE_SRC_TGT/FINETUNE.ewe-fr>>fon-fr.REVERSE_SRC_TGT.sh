#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=1024000M
#SBATCH --gpus=a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/fon-fr/%j_%x.out
#SBATCH --job-name=TEST.fon-fr.FINETUNE.ewe-fr>>fon-fr
#SBATCH --qos=cs
#SBATCH --partition=cs


python NMT/clean_slurm_outputs.py

nvidia-smi

# tensorboard --logdir "{tb_dir}" --port 6006 --host 0.0.0.0 &
srun python NMT/train.py \
	--config "/home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/fon-fr/FINETUNE.ewe-fr>>fon-fr.yaml" \
	--mode TEST \
	--REVERSE_SRC_TGT


python NMT/clean_slurm_outputs.py
