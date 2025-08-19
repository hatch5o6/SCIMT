#!/bin/bash

set -e
echo "Starting log_cognate_dataset_sizes.sh--"
date
echo "---------------------------------------"

CONFIGS=/home/hatch5o6/Cognate/code/Pipeline/cfg/SC/*.cfg
for config in $CONFIGS
do
    echo "Pipeline/format_SC.sh ${config}"
    bash "Pipeline/format_SC.sh" "${config}"
done

echo "Ending log_cognate_dataset_sizes.sh----"
date
echo "---------------------------------------"