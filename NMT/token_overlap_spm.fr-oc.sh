#!/bin/bash

#SBATCH --time=24:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=64000M
#SBATCH --gpus=0
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user thebrendanhatch@gmail.com
#SBATCH --output /home/hatch5o6/Cognate/code/NMT/slurm_outputs/%j_%x.out
#SBATCH --job-name=token_overlap_spm.fr-oc.out
#SBATCH --qos matrix



# # es-an
# python NMT/token_overlap_spm.py \
#     --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/es-en/train.csv \
#     --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es-an_en/es-an_en/es-an_en \
#     --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/an-en/train.csv \
#     --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es-an_en/es-an_en/es-an_en \
#     --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_es-an_feb17.json

# # es'-an
# python NMT/token_overlap_spm.py \
#     --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/SC/SC_es2an-en/train.csv \
#     --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an-an_en/SC_es2an-an_en/SC_es2an-an_en \
#     --sc_model_id_1 ES-AN-RNN-0-RNN-213 \
#     --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/an-en/train.csv \
#     --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an-an_en/SC_es2an-an_en/SC_es2an-an_en \
#     --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_es\'-an_feb17.json
# # good ^




# # fr-mfe
# python NMT/token_overlap_spm.py \
#     --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/fr-en/train.csv \
#     --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en/fr-mfe_en/fr-mfe_en \
#     --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/mfe-en/train.csv \
#     --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en/fr-mfe_en/fr-mfe_en \
#     --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_fr-mfe_feb17.json

# # fr'-mfe
# python NMT/token_overlap_spm.py \
#     --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/SC/SC_fr2mfe-en/train.csv \
#     --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en \
#     --sc_model_id_1 FR-MFE-RNN-0-RNN-102 \
#     --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/mfe-en/train.csv \
#     --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en \
#     --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_fr\'-mfe_feb17.json
# # good ^



# fr-oc
python NMT/token_overlap_spm.py \
    --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/fr-en/train.csv \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-oc_en/fr-oc_en/fr-oc_en \
    --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/oc-en/train.csv \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-oc_en/fr-oc_en/fr-oc_en \
    --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_fr-oc_feb17.json

# fr'-oc
python NMT/token_overlap_spm.py \
    --data1 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/SC/SC_fr2oc-en/train.csv \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2oc-oc_en/SC_fr2oc-oc_en/SC_fr2oc-oc_en \
    --sc_model_id_1 FR-OC-RNN-0-RNN-251 \
    --data2 /home/hatch5o6/Cognate/code/NMT/data/CharLOTTE/PLAIN/oc-en/train.csv \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2oc-oc_en/SC_fr2oc-oc_en/SC_fr2oc-oc_en \
    --out /home/hatch5o6/Cognate/code/NMT/results/vocab_overlap_fr\'-oc_feb17.json
