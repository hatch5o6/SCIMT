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
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%x.out
#SBATCH --job-name=NMT.SC_es2an-an.en
#SBATCH --qos=dw87

rm -r /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/NMT.SC_es2an.en_TRIAL_s=1000
python train.py \
    --config configs/NMT.SC_es2an-an.en.yaml \
    --mode TRAIN