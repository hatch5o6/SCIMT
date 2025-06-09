#!/bin/bash
set -e

source activate sound

HR_LANG=NGfr
LR_LANG=NGmfe
CHECKPOINT=checkpoint13

FILE=/home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/workspace/reference_models/bilingual/data/inference
if [ -d $FILE ]; then
    echo "deleting ${FILE}"
    rm -r $FILE
fi
FILE=/home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/workspace/reference_models/bilingual/rnn_${HR_LANG}-${LR_LANG}/0/results/inference_${CHECKPOINT}_${HR_LANG}_${LR_LANG}.${HR_LANG}
if [ -d $FILE ]; then
    echo "deleting ${FILE}"
    rm -r $FILE
fi
sleep 5

TRAIN_DATA=/home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/inputs/split_data/${HR_LANG}_${LR_LANG}/0
DATA=/home/hatch5o6/nobackup/archive/data/Kreyol-MT/bren-dan_TEST/test.bren
COPPER_MT_PREP_OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/inputs/split_data/${HR_LANG}_${LR_LANG}/inference

if [ -d $COPPER_MT_PREP_OUT_DIR ]; then
    echo "deleting ${COPPER_MT_PREP_OUT_DIR}"
    rm -r $COPPER_MT_PREP_OUT_DIR
fi
mkdir $COPPER_MT_PREP_OUT_DIR
FINAL_RESULTS=/home/hatch5o6/nobackup/archive/data/Kreyol-MT/bren-dan_TEST/test.bren.NGfr2NGmfe_model.hyp.txt

python hr_CopperMT.py \
    --data $DATA \
    --out $COPPER_MT_PREP_OUT_DIR \
    -hr $HR_LANG \
    -lr $LR_LANG \
    --training_data $TRAIN_DATA

dir_to_delete=/home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/workspace/reference_models/bilingual/rnn_${HR_LANG}-${LR_LANG}/0/results/inference_${CHECKPOINT}_${HR_LANG}_${LR_LANG}.${LR_LANG}
if [ -d $dir_to_delete ]
then
    echo "deleting ${dir_to_delete}"
    rm -r $dir_to_delete
fi

# run CopperMT model
source activate copper
cd /home/hatch5o6/Cognate/code/CopperMT/CopperMT/pipeline
sbatch main_nmt_bilingual_full_${HR_LANG}_${LR_LANG}.pred.sh --wait
sleep 90 # should be done predicting by the time this up

# join back into sentences
cd /home/hatch5o6/Cognate/code/NMT
source activate sound
python hr_CopperMT.py \
    --function retrieve \
    --data $DATA \
    --CopperMT_results /home/hatch5o6/nobackup/archive/CopperMT/${HR_LANG}_${LR_LANG}/workspace/reference_models/bilingual/rnn_${HR_LANG}-${LR_LANG}/0/results/inference_${CHECKPOINT}_${HR_LANG}_${LR_LANG}.${LR_LANG}/generate-test.txt \
    -hr $HR_LANG \
    -lr $LR_LANG \
    --out $FINAL_RESULTS 
