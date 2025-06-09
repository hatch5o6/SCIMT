#!/bin/bash

source $1
export WK_DIR INPUTS_DIR DATA_NAME langs

SELECTED_CHKPT=$2
seed=$3
predict_type=$4 #'inference' or 'test' # replace 'inference' throughout script with this variable
if [ $predict_type = 'test' ]
then
    echo "in main_nmt_bilingual_full_brendan_PREDICT.sh, predict_type='test', setting to seed ${seed}"
    predict_type=$seed
    PRED_TYPE=test
else
    echo "in main_nmt_bilingual_full_brendan_PREDICT.sh, predict_type='${predict_type}'"
    PRED_TYPE=$predict_type
fi
echo "    predict_type=${predict_type}"

nbest=$5
echo "    predicting nbest ${nbest}"
beam=$6
echo "    predicting beam ${beam}"


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo DIR is $DIR
USER_DIR="${DIR}/neural_translation/multilingual_rnns"  # Link to "multilingual_rnns"
echo USER_DIR is $USER_DIR

# ------ PARAMETERS
# INPUTS
ORIGIN_DATA_DIR="${INPUTS_DIR}/split_data/${DATA_NAME}"  # Link to the original data files
PARAMETER_DIR="${INPUTS_DIR}/parameters/bilingual_default"  # Contains the parameter files
# WORKING
DATA_DIR="${WK_DIR}/reference_models/bilingual/data"  # Where the data will be saved
WORK_DIR="${WK_DIR}/reference_models/bilingual"  # Where the models will be saved
mkdir -p "${WK_DIR}/reference_models/bilingual"
echo "${ORIGIN_DATA_DIR} ${PARAMETER_DIR} ${DATA_DIR} ${WORK_DIR}"

bash "${DIR}/neural_translation/data_preprocess.sh" \
    -l $lang \
    -o "${ORIGIN_DATA_DIR}/${predict_type}"\
    -d "${DATA_DIR}/${predict_type}"

# ------- TRAINING RNN AND TRANSFORMER
for lang_pairs in $lang ; do
    for model in "rnn"; do
        bash "${DIR}/neural_translation/inference_checkpoint.sh" \
            -l ${lang_pairs} -r ${lang_pairs} \
            -w "${WORK_DIR}/${model}_${lang_pairs}/${seed}" \
            -d "${DATA_DIR}/${predict_type}" \
            -u "${USER_DIR}" \
            -s "${SELECTED_CHKPT}" \
            -p "${PRED_TYPE}" \
            -n "${nbest}" \
            -b "${beam}"
    done
done