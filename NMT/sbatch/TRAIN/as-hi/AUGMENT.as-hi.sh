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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/as-hi/%j_%x.out
#SBATCH --job-name=TRAIN.as-hi.AUGMENT.as-hi
#SBATCH --qos=dw87

python NMT/clean_slurm_outputs.py

nvidia-smi
srun python NMT/train.py \
	--config /home/hatch5o6/Cognate/code/NMT/configs/CONFIGS/as-hi/AUGMENT.as-hi.yaml \
	--mode TRAIN


python NMT/clean_slurm_outputs.py
