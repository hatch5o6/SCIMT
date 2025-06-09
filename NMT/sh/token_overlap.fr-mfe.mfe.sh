python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe \
    --spm1  /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr.mfe/fr.mfe \
    --data2 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.fra \
    --spm2  /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr.mfe/fr.mfe \
    --is_parallel \
    --out sh/token_overlap.fr-mfe.mfe.parallel.json

python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr.mfe/fr.mfe \
    --data2 /home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.txt \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr.mfe/fr.mfe \
    --out sh/token_overlap.fr-mfe.mfe.json
