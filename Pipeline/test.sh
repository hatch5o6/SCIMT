#!/bin/bash

#### ARGUMENTS ####
echo "ECHOING" $1
source $1
export WK_DIR INPUTS_DIR
echo "WK_DIR ${WK_DIR}"
echo "INPUTS_DIR ${INPUTS_DIR}"

no_grouping=false
COGNATE_THRESH=0.5
SC_MODEL_TYPE=RNN
while getopts "s:t:c:gm:" opt; do
    case ${opt} in 
        s)  SRC=$OPTARG;;
        t)  TGT=$OPTARG;;
        c)  COG_DET_DATA=$OPTARG;;
        g)  no_grouping=true;;
        m)  SC_MODEL_TYPE=$OPTARG;;
    esac
done

echo SRC=$SRC
echo TGT=$TGT
echo COG_DET_DATA=$COG_DET_DATA
echo no_grouping=$no_grouping
echo SC_MODEL_TYPE=$SC_MODEL_TYPE

#### DETECT COGNATES ####
COGNATE_DIR=${COG_DET_DATA}/cognate
if [ -d $COGNATE_DIR ]
then
    echo "removing ${COGNATE_DIR}"
    rm -r $COGNATE_DIR
else
    echo "DOES NOT EXIST ${COGNATE_DIR}"
fi
echo "creating ${COGNATE_DIR}"
mkdir $COGNATE_DIR

FASTALIGN_DIR=${COG_DET_DATA}/fastalign
if [ -d $FASTALIGN_DIR ]
then
    echo "removing ${FASTALIGN_DIR}"
    rm -r $FASTALIGN_DIR
else
    echo "DOES NOT EXIST ${FASTALIGN_DIR}"
fi
echo "creating ${FASTALIGN_DIR}"
mkdir $FASTALIGN_DIR