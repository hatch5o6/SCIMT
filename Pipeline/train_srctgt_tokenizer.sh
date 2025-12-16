#!/bin/bash
set -e
echo "Starting-----------------------"
date
echo "-------------------------------"

# ARGUMENTS
config_file=$1
source $config_file # .cfg file from Pipeline/cfg/tok

echo "Arguments:-"
echo "    SPM_TRAIN_SIZE=$SPM_TRAIN_SIZE"
echo ""
echo "    SRC_LANGS=$SRC_LANGS"
echo "    SRC_TOK_NAME=$SRC_TOK_NAME"
echo "    TGT_LANGS=$TGT_LANGS"
echo "    TGT_TOK_NAME=$TGT_TOK_NAME"
echo ""
echo "    DIST=$DIST"
echo ""
echo "    TRAIN_PARALLEL=$TRAIN_PARALLEL"
# echo "    VAL_PARALLEL=$VAL_PARALLEL"
# echo "    TEST_PARALLEL=$TEST_PARALLEL"
echo "    TOK_TRAIN_DATA_DIR=$TOK_TRAIN_DATA_DIR"
echo "    SC_MODEL_ID=$SC_MODEL_ID"
echo "    VOCAB_SIZE=32000"
echo ""
echo "    SPLIT_ON_WS=$SPLIT_ON_WS"
echo "    INCLUDE_LANG_TOKS=$INCLUDE_LANG_TOKS" # TODO
echo "    INCLUDE_PAD_TOK=$INCLUDE_PAD_TOK"
echo "    SPECIAL_TOKS=$SPECIAL_TOKS" #TODO
echo "    IS_ATT=$IS_ATT"
echo "-------------------------------"


if [ -d $TOK_TRAIN_DATA_DIR ]
then
    echo "removing ${TOK_TRAIN_DATA_DIR}"
    rm -r $TOK_TRAIN_DATA_DIR
fi
echo "creating ${TOK_TRAIN_DATA_DIR}"
mkdir ${TOK_TRAIN_DATA_DIR}

SRCTGT_TOK_DIR=${TOK_TRAIN_DATA_DIR}/${SRC_TOK_NAME}_${TGT_TOK_NAME}

#TODO MAYBE INSTEAD PASS IN A LIST OF CSV FILES?
python Pipeline/make_tok_training_data.py \
    --train_csvs $TRAIN_PARALLEL \
    --out $TOK_TRAIN_DATA_DIR \
    --SC_MODEL_ID $SC_MODEL_ID \
    --IS_ATT $IS_ATT

# TODO Finish this if needed:
# USER_DEFINED_SYMBOLS
SRC_USER_DEFINED_SYMBOLS=()
TGT_USER_DEFINED_SYMBOLS=()

if [ $INCLUDE_PAD_TOK = true ]
then
    echo "INCLUDING <pad> TOKEN"
    SRC_USER_DEFINED_SYMBOLS+=('<pad>')
    TGT_USER_DEFINED_SYMBOLS+=('<pad>')
fi

# SRC_USER_DEFINED_SYMBOLS and TGT_USER_DEFINED_SYMBOLS should be the same.
IFS=',' read -r -a src_langs_array <<< $SRC_LANGS
IFS=',' read -r -a tgt_langs_array <<< $TGT_LANGS
langs_array=( "${src_langs_array[@]}" "${tgt_langs_array[@]}" )
if [ $INCLUDE_LANG_TOKS = true ]
then
    echo "INCLUDING LANG TOKENS"
    for l in ${langs_array[@]} ; do
        l_tok="<${l}>"
        echo "    $l_tok"
        SRC_USER_DEFINED_SYMBOLS+=(${l_tok})
        TGT_USER_DEFINED_SYMBOLS+=(${l_tok})
    done
fi

# SRC_USER_DEFINED_SYMBOLS and TGT_USER_DEFINED_SYMBOLS should be the same.
if [ $SPECIAL_TOKS != "null" ]
then
    echo "INCLUDING SPECIAL TOKENS: ${SPECIAL_TOKS}"
    IFS=',' read -r -a special_toks_array <<< $SPECIAL_TOKS
    SRC_USER_DEFINED_SYMBOLS=( "${SRC_USER_DEFINED_SYMBOLS[@]}" "${special_toks_array[@]}" )
    TGT_USER_DEFINED_SYMBOLS=( "${TGT_USER_DEFINED_SYMBOLS[@]}" "${special_toks_array[@]}" )
fi

SRC_USER_DEFINED_SYMBOLS_STR=""
for s in ${SRC_USER_DEFINED_SYMBOLS[@]} ; do
    SRC_USER_DEFINED_SYMBOLS_STR+="${s},"
done
if [[ $SRC_USER_DEFINED_SYMBOLS_STR ]];
then
    SRC_USER_DEFINED_SYMBOLS_STR=${SRC_USER_DEFINED_SYMBOLS_STR:0:-1}
else
    echo "no src user defined symbols"
fi
echo "SRC_USER_DEFINED_SYMBOLS_STR: ${SRC_USER_DEFINED_SYMBOLS_STR}"

TGT_USER_DEFINED_SYMBOLS_STR=""
for s in ${TGT_USER_DEFINED_SYMBOLS[@]} ; do
    TGT_USER_DEFINED_SYMBOLS_STR+="${s},"
done
if [[ $TGT_USER_DEFINED_SYMBOLS_STR ]];
then
    TGT_USER_DEFINED_SYMBOLS_STR=${TGT_USER_DEFINED_SYMBOLS_STR:0:-1}
else
    echo "no tgt user defined symbols"
fi
echo "TGT_USER_DEFINED_SYMBOLS_STR: ${TGT_USER_DEFINED_SYMBOLS_STR}"

# SRC_USER_DEFINED_SYMBOLS and TGT_USER_DEFINED_SYMBOLS should be the same.
if [ $SRC_USER_DEFINED_SYMBOLS_STR != $TGT_USER_DEFINED_SYMBOLS_STR ]
then
    echo "SRC_USER_DEFINED_SYMBOLS_STR should equal TGT_USER_DEFINED_SYMBOLS_STR"
    echo "Try Again. Exiting."
    exit
else
    echo "SRC_USER_DEFINED_SYMBOLS_STR == TGT_USER_DEFINED_SYMBOLS_STR"
    echo "as it should :)"
fi

echo ""
#### TRAIN TOKENIZERS ####
# # SRC TOKENIZER
# python NMT/spm_train.py \
#     --langs $SRC_LANGS \
#     --folder $TOK_TRAIN_DATA_DIR \
#     --training_data_size $SPM_TRAIN_SIZE \
#     --save_dir $SRC_TOK_DIR \
#     --spm_model_name $SRC_TOK_NAME \
#     --user_defined_symbols $SRC_USER_DEFINED_SYMBOLS_STR \
#     --SPLIT_ON_WS $SPLIT_ON_WS

# # TGT TOKENIZER
# python NMT/spm_train.py \
#     --langs $TGT_LANGS \
#     --folder $TOK_TRAIN_DATA_DIR \
#     --training_data_size $SPM_TRAIN_SIZE \
#     --save_dir $TGT_TOK_DIR \
#     --spm_model_name $TGT_TOK_NAME \
#     --user_defined_symbols $TGT_USER_DEFINED_SYMBOLS_STR \
#     --SPLIT_ON_WS $SPLIT_ON_WS

# SRCTGT TOKENIZER
python NMT/spm_train.py \
    --langs "${SRC_LANGS},${TGT_LANGS}" \
    --folder $TOK_TRAIN_DATA_DIR \
    --training_data_size $SPM_TRAIN_SIZE \
    --save_dir $SRCTGT_TOK_DIR \
    --spm_model_name "${SRC_TOK_NAME}_${TGT_TOK_NAME}" \
    --user_defined_symbols $SRC_USER_DEFINED_SYMBOLS_STR \
    --SPLIT_ON_WS $SPLIT_ON_WS \
    --spm_vocab_size $VOCAB_SIZE \
    --dist_str $DIST

# for l in ${langs_array[@]} ; do
#     echo "removing ${TOK_TRAIN_DATA_DIR}/${l}.txt"
#     rm ${TOK_TRAIN_DATA_DIR}/${l}.txt
# done

#Make SPM Training data for SC-applied data (Same process as Cognate Training data, but this time we're applying it to the the SC-processed data)

echo "Created by Cognate/code/Pipeline/train_tokenizer.sh ${config_file}" > ${TOK_TRAIN_DATA_DIR}/notes
date >> ${TOK_TRAIN_DATA_DIR}/notes

echo "Finished-----------------------"
date
echo "-------------------------------"