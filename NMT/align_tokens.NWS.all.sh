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
#SBATCH --job-name=align_tokens.NWS.all
#SBATCH --qos dw87

# python align_tokens.py \
#     --fr_file "/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.10M.txt" \
#     --sc_file "/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.10M.SC_FR-MFE-RNN-0_fr2mfe.txt" \
#     --sc_spm_name "/home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en" \
#     --TOTAL 500000

python align_tokens.py \
    --fr_file /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/training_data.s=1500div=fr.txt \
    --sc_file /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/training_data.s=1500div=fr.txt \
    --ch_file /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/training_data.s=1500div=mfe.txt \
    --tg_file /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr-mfe_en.NWS/fr-mfe_en/training_data.s=1500div=en.txt \
    --sc_spm_name /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe-mfe_en.NWS/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en \
    --write_collective_vocab /home/hatch5o6/nobackup/archive/CognateMT/spm_models/sc-aligned_fr-mfe_en.NWS \
    --fr_lang fr \
    --ch_lang mfe \
    --tg_lang en
    