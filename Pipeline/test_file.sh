
#### ARGUMENTS ####
source $1

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
echo "    COGNATE_THRESH=$COGNATE_THRESH"
echo "    COPPERMT_DATA_DIR=$COPPERMT_DATA_DIR"
echo "    COPPERMT_DIR=$COPPERMT_DIR"
echo "    PARAMETERS_DIR=$PARAMETERS_DIR"
echo "    RNN_HYPERPARAMS=$RNN_HYPERPARAMS"
echo "    VAL_COGNATES_SRC=$VAL_COGNATES_SRC"
echo "    VAL_COGNATES_TGT=$VAL_COGNATES_TGT"
echo "    TEST_COGNATES_SRC=$TEST_COGNATES_SRC"
echo "    TEST_COGNATES_TGT=$TEST_COGNATES_TGT"
echo "    COGNATE_TRAIN_RATIO=$COGNATE_TRAIN_RATIO"
echo "    COGNATE_TEST_RATIO=$COGNATE_TEST_RATIO"
echo "    COGNATE_VAL_RATIO=$COGNATE_VAL_RATIO"
echo "-------------------------------"









# str1="Hey, marshal"
# substr1="marshal"
# replace1="dallin"

# echo $str1
# echo $substr1
# echo $replace1

# newstr=${str1/$substr1/$replace1}

# echo "NEW STR: $newstr"

# PARALLEL_TRAIN=$1
# PARALLEL_VAL=$2
# PARALLEL_TEST=$3

# echo ""
# echo "ALLFILES:"
# PARALLEL_FILES=($PARALLEL_TRAIN $PARALLEL_VAL $PARALLEL_TEST)
# IFS=',' read -r -a APPLY_TO_FILES <<< $4
# ALL_CSV_FILES=( "${PARALLEL_FILES[@]}" "${APPLY_TO_FILES[@]}" )
# for f in ${ALL_CSV_FILES[@]} ; do
#     echo "    $f"
# done

# source $1
# export SC_MODEL_TYPE
# echo "SC_MODEL_TYPE $SC_MODEL_TYPE"

# echo "Training SC MODEL"
# echo "    TYPE=$SC_MODEL_TYPE"
# if [ $SC_MODEL_TYPE = "RNN" ]
# then
#     echo "(TEST)    ${COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan.sh ${PARAMETERS_F}"
# elif [ $SC_MODEL_TYPE = "SMT" ]
# then
#     echo "HEYO"
# fi

# echo "RUNNING COPPER MT"
# COPPERMT_DIR=/home/hatch5o6/Cognate/code/CopperMT/CopperMT
# PARAMETERS_F=/home/hatch5o6/Cognate/code/Pipeline/parameters/parameters.bren-dan.cfg
# bash "${COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan.sh" "${PARAMETERS_F}"