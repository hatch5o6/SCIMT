#!/bin/bash

set -e
echo "Starting-----------------------"
date
echo "-------------------------------"

#### ARGUMENTS ####
source $1 # .cfg file from from Pipeline/cfg/SC

PARAMETERS_F="${PARAMETERS_DIR}/parameters.${SRC}-${TGT}.cfg"
python Pipeline/write_scripts.py \
    --src ${SRC} \
    --tgt ${TGT} \
    --parameters $PARAMETERS_F

# select SC model
source activate copper
echo "-- Selecting SC MODEL --"
echo "    TYPE=$SC_MODEL_TYPE"
if [ $SC_MODEL_TYPE = "RNN" ]
then
    # select best model
    WORKSPACE_SEED_DIR=$COPPERMT_DATA_DIR/${SRC}_${TGT}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}
    SELECTED_RNN_CHECKPOINT=${WORKSPACE_SEED_DIR}/checkpoints/selected.pt
    if [ -f $SELECTED_RNN_CHECKPOINT ]
    then
        echo "    deleting $SELECTED_RNN_CHECKPOINT}"
        rm $SELECTED_RNN_CHECKPOINT
    fi
    echo "   python Pipeline/select_checkpoint.py --dir ${WORKSPACE_SEED_DIR}"
    python Pipeline/select_checkpoint.py --dir $WORKSPACE_SEED_DIR
# elif [ $SC_MODEL_TYPE = "SMT" ]
# then
#     # train SMT
#     cd ${COPPERMT_DIR}/pipeline
#     echo "    bash ${COPPERMT_DIR}/pipeline/main_smt_full_brendan.sh ${PARAMETERS_F} ${SEED}"
#     bash "${COPPERMT_DIR}/pipeline/main_smt_full_brendan.sh" "${PARAMETERS_F}" "${SEED}"
# # else
# #     echo "    INVALID SC_MODEL_TYPE: '${SC_MODEL_TYPE}'"
# #     exit
fi

INFERENCE_DATA_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}/workspace/reference_models/bilingual/data/inference
INFERENCE_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/inference_selected_checkpoint_${SRC}_${TGT}.${SRC}
for directory in $INFERENCE_DATA_DIR $INFERENCE_DIR ; do
    if [ -d $directory ]; then
        echo "    deleting ${directory}"
        rm -r $directory
    fi
done

#### APPLY SC ####
cd $MODULE_HOME_DIR
source activate sound
SPLIT_DATA=${COPPERMT_DATA_DIR}/${SRC}_${TGT}/inputs/split_data/${SRC}_${TGT}/${SEED}
COPPER_MT_PREP_OUT_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}/inputs/split_data/${SRC}_${TGT}/inference
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
            source activate copper
            echo "    main_nmt_bilingual_full_brendan_PREDICT.sh ${PARAMETERS_F} ${SELECTED_RNN_CHECKPOINT} ${SEED}"
            bash "main_nmt_bilingual_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${SELECTED_RNN_CHECKPOINT}" "${SEED}"
            COPPERMT_RESULTS=${COPPERMT_DATA_DIR}/${SRC}_${TGT}/workspace/reference_models/bilingual/rnn_${SRC}-${TGT}/${SEED}/results/inference_selected_checkpoint_${SRC}_${TGT}.${TGT}/generate-test.txt

            source activate sound
            cd $MODULE_HOME_DIR
            python NMT/hr_CopperMT.py \
                --function retrieve \
                --data $f \
                --CopperMT_results $COPPERMT_RESULTS \
                -hr $SRC \
                -lr $TGT
        elif [ $SC_MODEL_TYPE = "SMT" ]
        then
            source activate copper
            TEXT=$COPPER_MT_PREP_OUT_DIR/test_${SRC}_${TGT}.${SRC}
            HYP_OUT=$COPPER_MT_PREP_OUT_DIR/test_${SRC}_${TGT}.${TGT}
            echo "    main_smt_full_brendan_PREDICT.sh ${PARAMETERS_F} ${TEXT} ${HYP_OUT} ${SEED}"
            bash "main_smt_full_brendan_PREDICT.sh" "${PARAMETERS_F}" "${TEXT}" "${HYP_OUT}" "${SEED}"
            
            HYP_OUT_F=$HYP_OUT.hyp.txt
            source activate sound
            cd $MODULE_HOME_DIR
            python NMT/hr_CopperMT.py \
                --function retrieve \
                --data $f \
                --CopperMT_SMT_results ${TEXT},${HYP_OUT_F} \
                -hr $SRC \
                -lr $TGT
        fi
    fi
done


echo "Finished-----------------------"
date
echo "-------------------------------"