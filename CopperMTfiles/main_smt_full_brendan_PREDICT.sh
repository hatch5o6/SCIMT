#!/bin/bash
source $1
export WK_DIR INPUTS_DIR MOSES_DIR DATA_NAME lang

TEXT=$2
OUT=$3
seed=$4

HYP=${OUT}.hyp.txt
SAVE_N_BEST=${OUT}.10_best.txt

DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Where the data will be saved
WORK_DIR="${WK_DIR}/reference_models/statistical"  # Where the models will be saved

for lang_pair in $(echo ${lang} | tr "," "\n"); do
    IFS="-" read l_in l_out <<< "${lang_pair}";
    bash statistical_translation/model_inference.sh \
        -i "${l_in}" -o "${l_out}" \
        -m "${MOSES_DIR}" -w "${WORK_DIR}/${seed}" \
        -n 10 \
        -t ${TEXT} \
        -h ${HYP} \
        -b ${SAVE_N_BEST}
done

