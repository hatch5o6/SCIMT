#!/bin/bash

set -e
echo "Starting train_SC.sh-----------"
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

cd $MODULE_HOME_DIR
source activate sound
#### DETECT COGNATES ####
echo ""
echo ""
echo "#### DETECT COGNATES ####"
COGNATE_TRAIN=${COGNATE_TRAIN}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}
if [ -d $COGNATE_TRAIN ]
then
    echo "removing ${COGNATE_TRAIN}"
    rm -r $COGNATE_TRAIN
fi
echo "creating ${COGNATE_TRAIN}"
mkdir $COGNATE_TRAIN


COGNATE_DIR=${COGNATE_TRAIN}/cognate
if [ -d $COGNATE_DIR ]
then
    echo "removing ${COGNATE_DIR}"
    rm -r $COGNATE_DIR
fi
echo "creating ${COGNATE_DIR}"
mkdir $COGNATE_DIR

FASTALIGN_DIR=${COGNATE_TRAIN}/fastalign
if [ -d $FASTALIGN_DIR ]
then
    echo "removing ${FASTALIGN_DIR}"
    rm -r $FASTALIGN_DIR
fi
echo "creating ${FASTALIGN_DIR}"
mkdir $FASTALIGN_DIR

SRC_F=${COGNATE_DIR}/train.${SRC}
TGT_F=${COGNATE_DIR}/train.${TGT}

echo ""
# --src and --tgt are for filtering on $SRC and $TGT language pairs (the langauges we want cognates for)
python Pipeline/make_SC_training_data.py \
    --train_csv $PARALLEL_TRAIN \
    --val_csv $PARALLEL_VAL \
    --test_csv $PARALLEL_TEST \
    --src_out $SRC_F \
    --tgt_out $TGT_F \
    --src $SRC \
    --tgt $TGT
# cat ${COGNATE_TRAIN}/test.${SRC} ${COGNATE_TRAIN}/val.${SRC} ${COGNATE_TRAIN}/train.${SRC} > ${SRC_F}
# cat ${COGNATE_TRAIN}/test.${TGT} ${COGNATE_TRAIN}/val.${TGT} ${COGNATE_TRAIN}/train.${TGT} > ${TGT_F}

SRC_TGT_F=${FASTALIGN_DIR}/${SRC}-${TGT}
if [ $NO_GROUPING = true ]
then
    echo "NO_GROUPING has been set to true."
    WORD_LIST_OUT=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.NG.txt
    WORD_LIST_SRC=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.NG.cognates.${COGNATE_THRESH}.parallel-${SRC}.txt
    WORD_LIST_TGT=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.NG.cognates.${COGNATE_THRESH}.parallel-${TGT}.txt
    echo \t$WORD_LIST_OUT
else
    echo "NO_GROUPING has been set to false."
    WORD_LIST_OUT=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.txt
    WORD_LIST_SRC=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.cognates.${COGNATE_THRESH}.parallel-${SRC}.txt
    WORD_LIST_TGT=${FASTALIGN_DIR}/word_list.${SRC}-${TGT}.cognates.${COGNATE_THRESH}.parallel-${TGT}.txt
    echo \t$WORD_LIST_OUT
fi

if [ $REVERSE_SRC_TGT_COGNATES = true ]
then
    # prepare
    echo "prepare_fast_align.py REVERSE=true"
    python word_alignments/prepare_for_fastalign.py \
        --src $TGT_F \
        --tgt $SRC_F \
        --out ${SRC_TGT_F}.txt
    echo ""
else
    # prepare
    echo "prepare_fast_align.py REVERSE=false"
    python word_alignments/prepare_for_fastalign.py \
        --src $SRC_F \
        --tgt $TGT_F \
        --out ${SRC_TGT_F}.txt
    echo ""
fi

# fast_align
./../fast_align/build/fast_align \
    -i ${SRC_TGT_F}.txt \
    -d -o -v > ${SRC_TGT_F}.forward.align

./../fast_align/build/fast_align \
    -i ${SRC_TGT_F}.txt \
    -d -o -v -r > ${SRC_TGT_F}.reverse.align

./../fast_align/build/atools \
    -i ${SRC_TGT_F}.forward.align \
    -j ${SRC_TGT_F}.reverse.align \
    -c grow-diag-final-and > ${SRC_TGT_F}.sym.align
echo ""
echo ""

# make word alignments
if [ $NO_GROUPING = true ]
then
    echo "NO_GROUPING=true: running make_word_alignments_no_grouping.py"
    python word_alignments/make_word_alignments_no_grouping.py \
        --alignments ${SRC_TGT_F}.sym.align \
        --sent_pairs ${SRC_TGT_F}.txt \
        --out $WORD_LIST_OUT
else
    echo "NO_GROUPING=false: running make_word_alignments.py"
    python word_alignments/make_word_alignments.py \
    --alignments ${SRC_TGT_F}.sym.align \
    --sent_pairs ${SRC_TGT_F}.txt \
    --out $WORD_LIST_OUT
fi

if [ $REVERSE_SRC_TGT_COGNATES = true ]
then
    # get cognates
    echo "make_cognate_list.py REVERSE=true"
    python word_alignments/make_cognate_list.py \
        --word_list $WORD_LIST_OUT \
        --theta $COGNATE_THRESH \
        --src ${TGT} \
        --tgt ${SRC}
else
    # get cognates
    echo "make_cognate_list.py REVERSE=false"
    python word_alignments/make_cognate_list.py \
        --word_list $WORD_LIST_OUT \
        --theta $COGNATE_THRESH \
        --src ${SRC} \
        --tgt ${TGT}
fi


#### TRAIN SC MODEL WITH COPPER MT ####
echo \n\n
echo "#### TRAIN SC MODEL WITH COPPER MT ####"

# if needed, make datasets splits
if [ $TEST_COGNATES_SRC = "null" ]
then
    echo "Splitting cognate data ${WORD_LIST_SRC}, ${WORD_LIST_TGT}" 
    echo "    (train:val:test) ${COGNATE_TRAIN_RATIO}:${COGNATE_VAL_RATIO}:${COGNATE_TEST_RATIO}"
    TRAIN_COGNATES_SRC=${WORD_LIST_SRC:0:-3}train-s=${SEED}.txt
    TRAIN_COGNATES_TGT=${WORD_LIST_TGT:0:-3}train-s=${SEED}.txt

    TEST_COGNATES_SRC=${WORD_LIST_SRC:0:-3}test-s=${SEED}.txt
    TEST_COGNATES_TGT=${WORD_LIST_TGT:0:-3}test-s=${SEED}.txt

    VAL_COGNATES_SRC=${WORD_LIST_SRC:0:-3}val-s=${SEED}.txt
    VAL_COGNATES_TGT=${WORD_LIST_TGT:0:-3}val-s=${SEED}.txt

    python Pipeline/split.py \
        --data1 $WORD_LIST_SRC \
        --data2 $WORD_LIST_TGT \
        --train $COGNATE_TRAIN_RATIO \
        --val $COGNATE_VAL_RATIO \
        --test $COGNATE_TEST_RATIO \
        --seed $SEED \
        --out_dir $FASTALIGN_DIR \
        --UNIQUE_TEST
else
    TRAIN_COGNATES_SRC=$WORD_LIST_SRC
    TRAIN_COGNATES_TGT=$WORD_LIST_TGT
    echo "Reading train cognate data from $TRAIN_COGNATES_SRC, $TRAIN_COGNATES_TGT"
    echo "Reading val cognate data from $VAL_COGNATES_SRC, $VAL_COGNATES_TGT"
    echo "Reading test cognate data from $TEST_COGNATES_SRC, $TEST_COGNATES_TGT"
fi

echo ""
if [ "$ADDITIONAL_TRAIN_COGNATES_SRC" != "null" ]
then
    if [ "$ADDITIONAL_TRAIN_COGNATES_TGT" = "null" ]
    then
        echo "If ADDITIONAL_TRAIN_COGNATES_SRC is not 'null', then ADDITIONAL_TRAIN_COGNATES_TGT must also not be null!!!"
        exit
    fi
    echo "ADDING ADDITIONAL TRAINING COGNATES"
    TRAIN_COGNATES_SRC+=",${ADDITIONAL_TRAIN_COGNATES_SRC}"
    TRAIN_COGNATES_TGT+=",${ADDITIONAL_TRAIN_COGNATES_TGT}"
fi

echo "Cognate training data for format_data.py is as follows:"
echo "    TRAIN_COGNATES_SRC=${TRAIN_COGNATES_SRC}"
echo "    TRAIN_COGNATES_TGT=${TRAIN_COGNATES_TGT}"
echo "    VAL_COGNATES_SRC=${VAL_COGNATES_SRC}"
echo "    VAL_COGNATES_TGT=${VAL_COGNATES_TGT}"
echo "    TEST_COGNATES_SRC=${TEST_COGNATES_SRC}"
echo "    TEST_COGNATES_TGT=${TEST_COGNATES_TGT}"
echo ""

# format for CopperMT
COPPER_DIR=${COPPERMT_DATA_DIR}/${SRC}_${TGT}_${SC_MODEL_TYPE}-${RNN_HYPERPARAMS_ID}_S-${SEED}
echo "COPPER_DIR: ${COPPER_DIR}"

echo ""
if [ -d $COPPER_DIR ]
then
    echo "deleting ${COPPER_DIR}"
    rm -r $COPPER_DIR
fi
echo "creating ${COPPER_DIR}"
mkdir $COPPER_DIR
mkdir ${COPPER_DIR}/inputs
mkdir ${COPPER_DIR}/inputs/split_data
mkdir ${COPPER_DIR}/inputs/parameters
mkdir ${COPPER_DIR}/inputs/parameters/bilingual_default

if [ -d $RNN_HYPERPARAMS ]
then
    # cp -r $RNN_HYPERPARAMS ${COPPER_DIR}/inputs/parameters/bilingual_default/default_parameters_rnn_${SRC}-${TGT}.txt
    python Pipeline/copy_rnn_hyperparams.py \
        --rnn_hyperparam_id $RNN_HYPERPARAMS_ID \
        --rnn_hyperparams_dir $RNN_HYPERPARAMS \
        --copy_to_path ${COPPER_DIR}/inputs/parameters/bilingual_default/default_parameters_rnn_${SRC}-${TGT}.txt
fi

FORMAT_OUT_DIR=${COPPER_DIR}/inputs/split_data/${SRC}_${TGT}

echo ""
python CopperMT/format_data.py \
    --src_data $TRAIN_COGNATES_SRC \
    --tgt_data $TRAIN_COGNATES_TGT \
    --src $SRC \
    --tgt $TGT \
    --out_dir $FORMAT_OUT_DIR \
    --prefix train \
    --seed $SEED

echo ""
python CopperMT/format_data.py \
    --src_data $VAL_COGNATES_SRC \
    --tgt_data $VAL_COGNATES_TGT \
    --src $SRC \
    --tgt $TGT \
    --out_dir $FORMAT_OUT_DIR \
    --prefix fine_tune \
    --seed $SEED

echo ""
python CopperMT/format_data.py \
    --src_data $TEST_COGNATES_SRC \
    --tgt_data $TEST_COGNATES_TGT \
    --src $SRC \
    --tgt $TGT \
    --out_dir $FORMAT_OUT_DIR \
    --prefix test \
    --seed $SEED

python Pipeline/cognate_dataset_log.py \
    -f $FORMAT_OUT_DIR/$SEED \
    -l ${SRC}-${TGT}

echo "Finished-----------------------"
date
echo "-------------------------------"