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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%x.out
#SBATCH --job-name=NMT.SC_es2an-an.en.500k.test
#SBATCH --qos=dw87

nvidia-smi
srun python train.py \
    --config configs/an_en/NMT.SC_es2an-an.en.500k.yaml \
    --mode TEST