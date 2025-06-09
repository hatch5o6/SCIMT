#!/bin/bash
set -e

source activate sound

FILE=/home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/data/inference
if [ -f $FILE ]; then
    rm -r $FILE
fi
FILE=/home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_checkpoint_best_es_an.an
if [ -f $FILE ]; then
    rm -r $FILE
fi
sleep 5

HR_LANG=es
LR_LANG=an
TRAIN_DATA=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/es_an/0
DATA=/home/hatch5o6/nobackup/archive/data/flores200_dataset/dev/spa_Latn.dev
COPPER_MT_PREP_OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/es_an/inference
rm -r $COPPER_MT_PREP_OUT_DIR
mkdir $COPPER_MT_PREP_OUT_DIR
FINAL_RESULTS=/home/hatch5o6/nobackup/archive/data/flores200_dataset/dev/spa_Latn.SC_es-an.dev

python hr_CopperMT.py \
    --data $DATA \
    --out $COPPER_MT_PREP_OUT_DIR \
    -hr $HR_LANG \
    -lr $LR_LANG \
    --training_data $TRAIN_DATA

# run CopperMT model
source activate copper
cd /home/hatch5o6/Cognate/code/CopperMT/CopperMT/pipeline
sbatch main_nmt_bilingual_full_${HR_LANG}_${LR_LANG}.pred.sh --wait
sleep 180 # should be done predicting by the time this up

# join back into sentences
cd /home/hatch5o6/Cognate/code/NMT
source activate sound
python hr_CopperMT.py \
    --function retrieve \
    --data $DATA \
    --CopperMT_results /home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_checkpoint_best_es_an.an/generate-test.txt \
    -hr $HR_LANG \
    -lr $LR_LANG \
    --out $FINAL_RESULTS 
