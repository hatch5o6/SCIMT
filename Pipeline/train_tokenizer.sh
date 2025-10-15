#!/bin/bash
set -e
echo "Starting-----------------------"
date
echo "-------------------------------"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
echo "    TRAIN_PARALLEL=$TRAIN_PARALLEL"
echo "    VAL_PARALLEL=$VAL_PARALLEL"
echo "    TEST_PARALLEL=$TEST_PARALLEL"
echo "    TOK_TRAIN_DATA_DIR=$TOK_TRAIN_DATA_DIR"
echo ""
echo "    SPLIT_ON_WS=$SPLIT_ON_WS"
echo "    INCLUDE_LANG_TOKS=$INCLUDE_LANG_TOKS" # TODO
echo "    INCLUDE_PAD_TOK=$INCLUDE_PAD_TOK"
echo "    SPECIAL_TOKS=$SPECIAL_TOKS" #TODO
echo "-------------------------------"

if [ -d $TOK_TRAIN_DATA_DIR ]
then
    echo "removing ${TOK_TRAIN_DATA_DIR}"
    rm -r $TOK_TRAIN_DATA_DIR
fi
echo "creating ${TOK_TRAIN_DATA_DIR}"
mkdir ${TOK_TRAIN_DATA_DIR}

SRC_TOK_DIR=${TOK_TRAIN_DATA_DIR}/${SRC_TOK_NAME}
TGT_TOK_DIR=${TOK_TRAIN_DATA_DIR}/${TGT_TOK_NAME}

#TODO MAYBE INSTEAD PASS IN A LIST OF CSV FILES?
# Set default values for SC_MODEL_ID and IS_ATT if not provided in config
: ${SC_MODEL_ID:=null}
: ${IS_ATT:=false}

python "${SCRIPT_DIR}/make_tok_training_data.py" \
    --train_csvs $TRAIN_PARALLEL \
    --val_csvs $VAL_PARALLEL  \
    --test_csvs $TEST_PARALLEL  \
    --out_dir $TOK_TRAIN_DATA_DIR \
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
# Remove trailing comma if string is not empty
if [[ -n "$SRC_USER_DEFINED_SYMBOLS_STR" ]];
then
    SRC_USER_DEFINED_SYMBOLS_STR="${SRC_USER_DEFINED_SYMBOLS_STR%,}"
else
    echo "no src user defined symbols"
fi
echo "SRC_USER_DEFINED_SYMBOLS_STR: ${SRC_USER_DEFINED_SYMBOLS_STR}"

TGT_USER_DEFINED_SYMBOLS_STR=""
for s in ${TGT_USER_DEFINED_SYMBOLS[@]} ; do
    TGT_USER_DEFINED_SYMBOLS_STR+="${s},"
done
# Remove trailing comma if string is not empty
if [[ -n "$TGT_USER_DEFINED_SYMBOLS_STR" ]];
then
    TGT_USER_DEFINED_SYMBOLS_STR="${TGT_USER_DEFINED_SYMBOLS_STR%,}"
else
    echo "no tgt user defined symbols"
fi
echo "TGT_USER_DEFINED_SYMBOLS_STR: ${TGT_USER_DEFINED_SYMBOLS_STR}"

echo ""
#### TRAIN TOKENIZERS ####

# Train a combined tokenizer for both source and target languages
# The CharLOTTE methodology uses a single shared tokenizer
COMBINED_TOK_NAME="${SRC_TOK_NAME}_${TGT_TOK_NAME}"
COMBINED_LANGS="${SRC_LANGS},${TGT_LANGS}"

# Determine output directory - check if VOCAB_SIZE is set (for spm_models location)
if [[ -n "${VOCAB_SIZE}" ]]; then
    # Save to spm_models directory (standard location for NMT)
    SPM_MODELS_DIR="$(dirname "${TOK_TRAIN_DATA_DIR}")/spm_models"
    COMBINED_TOK_DIR="${SPM_MODELS_DIR}/${COMBINED_TOK_NAME}"
else
    # Fallback to tokenizer_data subdirectory
    COMBINED_TOK_DIR="${TOK_TRAIN_DATA_DIR}/${COMBINED_TOK_NAME}"
fi

echo "Training combined tokenizer for: ${COMBINED_LANGS}"
echo "Tokenizer will be saved to: ${COMBINED_TOK_DIR}/${COMBINED_TOK_NAME}.model"

# Automatically detect which language files were created
# make_tok_training_data.py creates files based on actual language codes (e.g., es.txt, pt.txt, en.txt)
# not on logical names (e.g., es2pt.txt)
ACTUAL_LANGS=""
for txt_file in "${TOK_TRAIN_DATA_DIR}"/*.txt; do
    if [ -f "$txt_file" ]; then
        lang=$(basename "$txt_file" .txt)
        if [ -z "$ACTUAL_LANGS" ]; then
            ACTUAL_LANGS="$lang"
        else
            ACTUAL_LANGS="$ACTUAL_LANGS,$lang"
        fi
    fi
done

echo "Detected language files: ${ACTUAL_LANGS}"

# Create output directory
mkdir -p "${COMBINED_TOK_DIR}"

python "${SCRIPT_DIR}/../NMT/spm_train.py" \
    --langs "$ACTUAL_LANGS" \
    --folder "$TOK_TRAIN_DATA_DIR" \
    --dist_str "${DIST:-$COMBINED_LANGS}" \
    --training_data_size $SPM_TRAIN_SIZE \
    --save_dir "$COMBINED_TOK_DIR" \
    --spm_model_name "$COMBINED_TOK_NAME" \
    --spm_vocab_size "${VOCAB_SIZE:-8000}" \
    --user_defined_symbols "$SRC_USER_DEFINED_SYMBOLS_STR" \
    --SPLIT_ON_WS $SPLIT_ON_WS

echo "Tokenizer training complete!"
echo "Model files created in: ${COMBINED_TOK_DIR}"

# for l in ${langs_array[@]} ; do
#     echo "removing ${TOK_TRAIN_DATA_DIR}/${l}.txt"
#     rm ${TOK_TRAIN_DATA_DIR}/${l}.txt
# done

#Make SPM Training data for SC-applied data (Same process as Cognate Training data, but this time we're applying it to the the SC-processed data)


#### TRAIN NMT MODEL ####

echo "Created by Cognate/code/Pipeline/train_tokenizer.sh ${config_file}" > ${TOK_TRAIN_DATA_DIR}/notes
date >> ${TOK_TRAIN_DATA_DIR}/notes

echo "Finished-----------------------"
date
echo "-------------------------------"