#!/bin/bash
set -e
echo "Starting-----------------------"
date
echo "-------------------------------"

# ARGUMENTS

#### TRAIN TOKENIZERS ####

python NMT/spm_train.py \
    --data

#Make SPM Training data for SC-applied data (Same process as Cognate Training data, but this time we're applying it to the the SC-processed data)


#### TRAIN NMT MODEL ####

echo "Finished-----------------------"
date
echo "-------------------------------"