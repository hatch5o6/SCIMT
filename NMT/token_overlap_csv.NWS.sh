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
#SBATCH --job-name=token_overlap_csv.NWS.out
#SBATCH --qos dw87

python token_overlap_csv.py \
    --pl fr\
    --cl mfe\
    --tl en\
    --data_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN/mfe-en/train-testing.csv\
    --sc_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/SC/SC_fr2mfe-en/train-testing.csv\
    --sc_model_id FR-MFE-RNN-0\
    --og_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/fr-mfe_en\
    --sc_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en\
    --aligned_vocab /home/hatch5o6/nobackup/archive/CognateMT/spm_models/sc-aligned_fr-mfe_en.NWS/FINAL_COLLECTIVE_VOCAB.json\
    --VOCAB_SIZE_CAP 32000 \
    --out token_overlap_csv.scores.VOCAB_SIZE_CAP-32K.NWS.out


python token_overlap_csv.py \
    --pl fr\
    --cl mfe\
    --tl en\
    --data_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN/mfe-en/train-testing.csv\
    --sc_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/SC/SC_fr2mfe-en/train-testing.csv\
    --sc_model_id FR-MFE-RNN-0\
    --og_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/fr-mfe_en\
    --sc_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en\
    --aligned_vocab /home/hatch5o6/nobackup/archive/CognateMT/spm_models/sc-aligned_fr-mfe_en.NWS/FINAL_COLLECTIVE_VOCAB.json\
    --VOCAB_SIZE_CAP 16000 \
    --out token_overlap_csv.scores.VOCAB_SIZE_CAP-16K.NWS.out


python token_overlap_csv.py \
    --pl fr\
    --cl mfe\
    --tl en\
    --data_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN/mfe-en/train-testing.csv\
    --sc_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/SC/SC_fr2mfe-en/train-testing.csv\
    --sc_model_id FR-MFE-RNN-0\
    --og_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/fr-mfe_en\
    --sc_spm /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en\
    --aligned_vocab /home/hatch5o6/nobackup/archive/CognateMT/spm_models/sc-aligned_fr-mfe_en.NWS/FINAL_COLLECTIVE_VOCAB.json\
    --VOCAB_SIZE_CAP 64000 \
    --out token_overlap_csv.scores.VOCAB_SIZE_CAP-64K.NWS.out