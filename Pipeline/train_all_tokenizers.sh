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
#SBATCH --job-name=train_all_tokenizers
#SBATCH --qos dw87

rm -r /home/hatch5o6/nobackup/archive/CognateMT/spm_models/*
rm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes

set -e

for FILE in Pipeline/cfg/tok/*; do
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
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/bho_hi.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.bho_hi.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/bn-as_hi.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.bn-as_hi.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/cs-hsb_de.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.cs-hsb_de.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/djk_en.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.djk_en.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/es-an_en.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.es-an_en.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/ewe-fon_en.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.ewe-fon_en.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/ewe-fon_fr.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.ewe-fon_fr.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/fr-mfe_en.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.fr-mfe_en.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/hi-bho_as.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.hi-bho_as.out
# bash Pipeline/train_srctgt_tokenizer.sh Pipeline/cfg/tok/lua-bem_en.cfg > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_srctgt_tokenizer.lua-bem_en.out

echo "Created by Cognate/code/Pipeline/train_all_tokenizers.sh" > /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes
date >> /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes

python Pipeline/clean_slurm_outputs.py