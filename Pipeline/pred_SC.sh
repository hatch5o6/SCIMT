#!/bin/bash

set -e

__conda_setup="$('/vapps/rhel9/x86_64/miniconda3/latest/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

echo "Starting pred_SC.sh------------"
date
echo "-------------------------------"

#### ARGUMENTS ####
source $1 # .cfg file from from Pipeline/cfg/SC
echo "Arguments:-"
echo "    MODULE_HOME_DIR=$MODULE_HOME_DIR"
echo ""
echo "    SRC=$SRC"
echo "    TGT=$TGT"
echo "    PARALLEL_TRAIN=$PARALLEL_TRAIN"
echo "    PARALLEL_VAL=$PARALLEL_VAL"
echo "    PARALLEL_TEST=$PARALLEL_TEST"
echo "    COGNATE_TRAIN=$COGNATE_TRAIN"
echo "    NO_GROUPING=$NO_GROUPING"
echo "    SC_MODEL_TYPE=$SC_MODEL_TYPE"
echo "    SEED=$SEED"
echo "    SC_MODEL_ID=$SC_MODEL_ID"
echo "    COGNATE_THRESH=$COGNATE_THRESH"
echo "    COPPERMT_DATA_DIR=$COPPERMT_DATA_DIR"
echo "    COPPERMT_DIR=$COPPERMT_DIR"
echo "    PARAMETERS_DIR=$PARAMETERS_DIR"
echo "    RNN_HYPERPARAMS=$RNN_HYPERPARAMS"
echo "    RNN_HYPERPARAMS_ID=$RNN_HYPERPARAMS_ID"
echo "    BEAM=$BEAM"
echo "    NBEST=$NBEST"
echo "    REVERSE_SRC_TGT_COGNATES=$REVERSE_SRC_TGT_COGNATES"
echo "    VAL_COGNATES_SRC=$VAL_COGNATES_SRC"
echo "    VAL_COGNATES_TGT=$VAL_COGNATES_TGT"
echo "    TEST_COGNATES_SRC=$TEST_COGNATES_SRC"
echo "    TEST_COGNATES_TGT=$TEST_COGNATES_TGT"
echo "    COGNATE_TRAIN_RATIO=$COGNATE_TRAIN_RATIO"
echo "    COGNATE_TEST_RATIO=$COGNATE_TEST_RATIO"
echo "    COGNATE_VAL_RATIO=$COGNATE_VAL_RATIO"
echo "-------------------------------"

PARAMETERS_F="${PARAMETERS_DIR}/parameters.${SRC}-${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}.cfg"
python Pipeline/write_scripts.py \
    --src ${SRC} \
    --tgt ${TGT} \
    --coppermt_data_dir ${COPPERMT_DATA_DIR} \
    --sc_model_type ${SC_MODEL_TYPE} \
    --rnn_hyperparams_id ${RNN_HYPERPARAMS_ID} \
    --seed ${SEED} \
    --parameters $PARAMETERS_F

# select SC model
conda activate copper
echo "-- Getting selected SC MODEL --"
echo "    TYPE=$SC_MODEL_TYPE"
if [ $SC_MODEL_TYPE = "RNN" ]
then
    # # select best model -- doing this in the train_SC.sh file. Don't need, and in fact, should not, do this here.
    # WORKSPACE_SEED_DIR=$COPPERMT_DATA_DIR/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}
    # SELECTED_RNN_CHECKPOINT=${WORKSPACE_SEED_DIR}/checkpoints/selected.pt
    # if [ -f $SELECTED_RNN_CHECKPOINT ]
    # then
    #     echo "    deleting ${SELECTED_RNN_CHECKPOINT}"
    #     rm $SELECTED_RNN_CHECKPOINT
    # fi
    # echo "   python Pipeline/select_checkpoint.py --dir ${WORKSPACE_SEED_DIR}"
    # python Pipeline/select_checkpoint.py --dir $WORKSPACE_SEED_DIR

    # Get selected model
    WORKSPACE_SEED_DIR=$COPPERMT_DATA_DIR/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}
    SELECTED_RNN_CHECKPOINT=${WORKSPACE_SEED_DIR}/checkpoints/selected.pt
fi

INFERENCE_DATA_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/data/inference
INFERENCE_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/inference_selected_checkpoint_${SRC}_${TGT}.${SRC}
for directory in $INFERENCE_DATA_DIR $INFERENCE_DIR ; do
    if [ -d $directory ]; then
        echo "    deleting ${directory}"
        rm -r $directory
    fi
done

#### APPLY SC ####
cd $MODULE_HOME_DIR
conda activate sound
SPLIT_DATA=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/inputs/split_data/${SRC}_${TGT}/${SEED}
COPPER_MT_PREP_OUT_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/inputs/split_data/${SRC}_${TGT}/inference

if [ -d $COPPER_MT_PREP_OUT_DIR ]; then
    echo "deleting ${COPPER_MT_PREP_OUT_DIR}"
    rm -r $COPPER_MT_PREP_OUT_DIR
fi
mkdir $COPPER_MT_PREP_OUT_DIR

PARALLEL_FILES=( $PARALLEL_TRAIN $PARALLEL_VAL $PARALLEL_TEST )
IFS="," read -r -a APPLY_TO_FILES <<< $APPLY_TO
ALL_CSV_FILES=( "${PARALLEL_FILES[@]}" "${APPLY_TO_FILES[@]}" )
echo "-- APPLYING SC MODEL TO FILES --"
for f in ${ALL_CSV_FILES[@]} ; do
    if [ $f = "null" ]
    then
        echo "    DATA CSV F is null"
    else
        echo "    DATA CSV F: ${f}"
        echo ""
        python NMT/hr_CopperMT.py \
            --data $f \
            --out $COPPER_MT_PREP_OUT_DIR \
            -hr $SRC \
            -lr $TGT \
            --training_data $SPLIT_DATA \
            --limit_lang $SRC

        cd ${COPPERMT_DIR}/pipeline
        if [ $SC_MODEL_TYPE = "RNN" ]
        then
            conda activate copper
            echo "    main_nmt_bilingual_full_brendan_PREDICT.sh ${PARAMETERS_F} ${SELECTED_RNN_CHECKPOINT} ${SEED} inference ${NBEST} ${BEAM}"
            bash "main_nmt_bilingual_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${SELECTED_RNN_CHECKPOINT}" "${SEED}" "inference" "${NBEST}" "${BEAM}"
            COPPERMT_RESULTS=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/inference_selected_checkpoint_${SRC}_${TGT}.${TGT}/generate-test.txt

            conda activate sound
            cd $MODULE_HOME_DIR
            python NMT/hr_CopperMT.py \
                --function retrieve \
                --data $f \
                --CopperMT_results $COPPERMT_RESULTS \
                -hr $SRC \
                -lr $TGT \
                --MODEL_ID $SC_MODEL_ID
        elif [ $SC_MODEL_TYPE = "SMT" ]
        then
            conda activate copper
            TEXT=$COPPER_MT_PREP_OUT_DIR/test_${SRC}_${TGT}.${SRC}
            HYP_OUT=$COPPER_MT_PREP_OUT_DIR/test_${SRC}_${TGT}.${TGT}
            echo "    main_smt_full_brendan_PREDICT.sh ${PARAMETERS_F} ${TEXT} ${HYP_OUT} ${SEED}"
            bash "main_smt_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${TEXT}" "${HYP_OUT}" "${SEED}"
            
            HYP_OUT_F=$HYP_OUT.hyp.txt
            conda activate sound
            cd $MODULE_HOME_DIR
            python NMT/hr_CopperMT.py \
                --function retrieve \
                --data $f \
                --CopperMT_SMT_results ${TEXT},${HYP_OUT_F} \
                -hr $SRC \
                -lr $TGT \
                --MODEL_ID $SC_MODEL_ID
        fi
    fi
done


echo "Finished-----------------------"
date
echo "-------------------------------"