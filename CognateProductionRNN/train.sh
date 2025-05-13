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
#SBATCH --output /home/hatch5o6/Cognate/code/CognateProductionRNN/slurm_outputs/%x.out
#SBATCH --job-name=cognate_production_rnn_trial_0
#SBATCH --qos=dw87

rm -r /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/TRIAL_0
python train.py \
    --config configs/es2an.yaml