python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe \
    --spm1  /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe.mfe/SC_fr2mfe.mfe \
    --data2 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.SC_NGfr2NGmfe \
    --spm2  /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe.mfe/SC_fr2mfe.mfe \
    --is_parallel \
    --out sh/token_overlap.SC_fr2mfe.mfe.parallel.json

python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe.mfe/SC_fr2mfe.mfe \
    --data2 /home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.100k.SC_NGfr2NGmfe.txt \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_fr2mfe.mfe/SC_fr2mfe.mfe \
    --out sh/token_overlap.SC_fr2mfe.mfe.json
