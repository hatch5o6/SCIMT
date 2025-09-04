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
#SBATCH --job-name=assert_no_data_overlap.plain.augmented
#SBATCH --qos dw87



python assert_no_data_overlap.py \
    --dir /home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN \
    --PLAIN_dir /home/hatch5o6/Cognate/code/NMT/data/PLAIN > assert_no_data_overlap.augmented_data.PLAIN.matthew_tests.out
aug_output_content=$(<assert_no_data_overlap.augmented_data.PLAIN.matthew_tests.out)
# aug_gt_content=
# if [ "${aug_output_content}" = "${aug_gt_content}" ]
# then
#     echo "AUG TEST PASSED"
# else
#     echo "AUG TEST FAILED"
# fi


python clean_slurm_outputs.py
