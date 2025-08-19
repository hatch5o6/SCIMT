#!/bin/bash

# CFGS=/home/hatch5o6/Cognate/code/Pipeline/cfg/tok
# for f in "/home/hatch5o6/Cognate/code/Pipeline/cfg/tok/*.cfg";
# do
#     echo ${f}
#     echo "HEY THERE HELP"
#     # source $f
#     # echo "-----------------------"
#     # echo "SRC_LANGS: ${SRC_LANGS}"
#     # echo "    SRC_TOK_NAME: ${SRC_TOK_NAME}"
#     # echo "TGT_LANGS: ${TGT_LANGS}"
#     # echo "    TGT_TOK_NAME: ${TGT_TOK_NAME}"
#     # sh Pipeline/train_srctgt_tokenizer.sh $f > /home/hatch5o6/Cognate/code/Pipeline/sh_outputs/train_tokenizer.${SRC_TOK_NAME}_${TGT_TOK_NAME}.out
# done


rm -r /home/hatch5o6/nobackup/archive/CognateMT/spm_models/*
rm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/notes

for FILE in Pipeline/cfg/tok/*; do
    echo "    train_srctgt_tokenizer.sh ${FILE}"
    bash Pipeline/train_srctgt_tokenizer.sh $FILE > ${FILE}.run.out
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
date >> /home/hatch5o6/nobackup/archive/CognateMT/spm_models