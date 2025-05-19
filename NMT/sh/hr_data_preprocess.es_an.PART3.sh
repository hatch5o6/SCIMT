#!/bin/bash
set -e

HR_LANG=es
LR_LANG=an
TRAIN_DATA=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/es_an/0
DATA=/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.es
COPPER_MT_PREP_OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/es_an/inference
FINAL_RESULTS=/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.es.SC_es2an

# source activate sound
# python hr_CopperMT.py \
#     --data $DATA \
#     --out $COPPER_MT_PREP_OUT_DIR \
#     -hr $HR_LANG \
#     -lr $LR_LANG \
#     --training_data $TRAIN_DATA

# # run CopperMT model
# source activate copper
# sbatch /home/hatch5o6/Cognate/code/CopperMT/CopperMT/pipeline/main_nmt_bilingual_full_${HR_LANG}_${LR_LANG}.pred.sh
# wait
# sleep 10

# join back into sentences
source activate sound
python hr_CopperMT.py \
    --function retrieve \
    --data $DATA \
    --CopperMT_results /home/hatch5o6/nobackup/archive/CopperMT/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_checkpoint_best_es_an.an/generate-test.txt \
    -hr $HR_LANG \
    -lr $LR_LANG \
    --out $FINAL_RESULTS
