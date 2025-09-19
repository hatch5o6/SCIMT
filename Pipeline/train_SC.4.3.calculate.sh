#!/bin/bash

set -e

__conda_setup="$('/vapps/rhel9/x86_64/miniconda3/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

echo "Starting train_SC.sh-----------"
date
echo "-------------------------------"

    ###############################################
#-- #                 1) ARGUMENTS                # --#
    ###############################################
echo "    ###############################################"
echo "#-- #                 1) ARGUMENTS                # --#"
echo "    ###############################################"
source $1 # .cfg file from from Pipeline/cfg/SC

echo "Arguments:-"
echo "    MODULE_HOME_DIR=$MODULE_HOME_DIR"
echo ""
echo "    SRC=$SRC"
echo "    TGT=$TGT"
echo "    PARALLEL_TRAIN=$PARALLEL_TRAIN"
echo "    PARALLEL_VAL=$PARALLEL_VAL"
echo "    PARALLEL_TEST=$PARALLEL_TEST"
# echo "    APPLY_TO=$APPLY_TO" # not used
echo "    COGNATE_TRAIN=$COGNATE_TRAIN"
echo "    NO_GROUPING=$NO_GROUPING"
echo "    SC_MODEL_TYPE=$SC_MODEL_TYPE"
echo "    SEED=$SEED"
# echo "    SC_MODEL_ID=$SC_MODEL_ID" # not used
echo "    COGNATE_THRESH=$COGNATE_THRESH"
echo "    COPPERMT_DATA_DIR=$COPPERMT_DATA_DIR"
echo "    COPPERMT_DIR=$COPPERMT_DIR"
echo "    PARAMETERS_DIR=$PARAMETERS_DIR"
echo "    RNN_HYPERPARAMS=$RNN_HYPERPARAMS"
echo "    RNN_HYPERPARAMS_ID=$RNN_HYPERPARAMS_ID"
echo "    BEAM=$BEAM"
echo "    NBEST=$NBEST"
echo "    REVERSE_SRC_TGT_COGNATES=$REVERSE_SRC_TGT_COGNATES"
echo "    ADDITIONAL_TRAIN_COGNATES_SRC=$ADDITIONAL_TRAIN_COGNATES_SRC"
echo "    ADDITIONAL_TRAIN_COGNATES_TGT=$ADDITIONAL_TRAIN_COGNATES_TGT"
echo "    VAL_COGNATES_SRC=$VAL_COGNATES_SRC"
echo "    VAL_COGNATES_TGT=$VAL_COGNATES_TGT"
echo "    TEST_COGNATES_SRC=$TEST_COGNATES_SRC"
echo "    TEST_COGNATES_TGT=$TEST_COGNATES_TGT"
echo "    COGNATE_TRAIN_RATIO=$COGNATE_TRAIN_RATIO"
echo "    COGNATE_TEST_RATIO=$COGNATE_TEST_RATIO"
echo "    COGNATE_VAL_RATIO=$COGNATE_VAL_RATIO"
echo "-------------------------------"

cd ${COPPERMT_DIR}/pipeline
SPLIT_DATA=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/inputs/split_data/${SRC}_${TGT}/${SEED}

echo "Testing model"
if [ $SC_MODEL_TYPE = "RNN" ]
then
    # echo "    main_nmt_bilingual_full_brendan_PREDICT.sh ${PARAMETERS_F} ${SELECTED_RNN_CHECKPOINT} ${SEED} test ${NBEST} ${BEAM}"
    # bash "main_nmt_bilingual_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${SELECTED_RNN_CHECKPOINT}" "${SEED}" "test" "${NBEST}" "${BEAM}"
    HYP_OUT_TXT=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/test_selected_checkpoint_${SRC}_${TGT}.${TGT}/generate-test.txt
    TEST_OUT_F=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/test_selected_checkpoint_${SRC}_${TGT}.${TGT}/generate-test.hyp.txt
    
    # # Go back to normal directory
    # cd $MODULE_HOME_DIR
    # conda activate sound
    # python NMT/hr_CopperMT.py \
    #     --function get_test_results \
    #     --test_src $SRC_TEXT \
    #     --data $HYP_OUT_TXT \
    #     --out $TEST_OUT_F

    SCORES_OUT_F=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/test_selected_checkpoint_${SRC}_${TGT}.${TGT}/generate-test.hyp.scores.txt
elif [ $SC_MODEL_TYPE = "SMT" ]
then
    # HYP_OUT=${SPLIT_DATA}/test_${SRC}_${TGT}.${TGT}
    # echo "    main_smt_full_brendan_PREDICT.sh ${PARAMETERS_F} ${SRC_TEXT} ${HYP_OUT} ${SEED}"
    # bash "main_smt_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${SRC_TEXT}" "${HYP_OUT}" "${SEED}"
    TEST_OUT_F=$HYP_OUT.hyp.txt    
    SCORES_OUT_F=$HYP_OUT.hyp.scores.txt
    
    # Go back to normal directory
    cd $MODULE_HOME_DIR
    conda activate sound
fi


# 4.3 Calculate scores
conda activate sound
echo ""
echo ""
echo "######## 4.3 Calculate scores ########"
REF_TEXT=${SPLIT_DATA}/test_${SRC}_${TGT}.${TGT}
echo "Calculating Scores"
echo "    NMT/evaluate.py --ref ${REF_TEXT} --hyp ${TEST_OUT_F} --out ${SCORES_OUT_F}"
python NMT/evaluate.py --ref ${REF_TEXT} --hyp ${TEST_OUT_F} --out ${SCORES_OUT_F}
echo "    scores written to ${SCORES_OUT_F}"
cat ${SCORES_OUT_F}