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
#SBATCH --output /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/slurm_outputs/%j_%x.out
#SBATCH --job-name=train_oc_abl_tokenizers
#SBATCH --qos matrix

set -e

for FILE in Pipeline/cfg/tok_oc_abl/*; do
    if [ $FILE == "Pipeline/cfg/tok_oc_abl/archive"  ]
    then
        continue
    fi

    if [[ $FILE == *"esx"* ]]
    then
        continue
    fi
    
    echo "##################################################################################################################################"
    echo "    train_srctgt_tokenizer.sh ${FILE}"
    bash Pipeline/train_srctgt_tokenizer.sh $FILE
    echo "Finished Tokenizer-------------"
    echo "(${FILE})"
    date
    echo "-------------------------------"


    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
    echo ""
done

echo "Created by Cognate/code/Pipeline/train_oc_abl_tokenizers.sh" > /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes_oc_abl
date >> /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes_oc_abl

python Pipeline/clean_slurm_outputs.py