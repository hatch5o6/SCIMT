#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%j_%x.out
#SBATCH --job-name=assert_no_data_overlap.make_no_overlap.arabic
#SBATCH --qos dw87
set -e

TAG=no_overlap_v1

# Run tests and make fixes (make it so train, val, and test have no overlapping src and tgt lines by removing duplicate lines from training and then dev)
python assert_no_data_overlap.py \
    --dir /home/hatch5o6/Cognate/code/NMT/data/arabic \
    --INCLUDES_DEV_TEST \
    --MAKE_FIXES \
    --MAKE_FIXES_TAG $TAG

# Assert that the newly created data (from previous command ^) passes tests
python assert_no_data_overlap.py \
    --dir /home/hatch5o6/Cognate/code/NMT/data/arabic \
    --HAVE_TAG $TAG \
    --INCLUDES_DEV_TEST

python clean_slurm_outputs.py

