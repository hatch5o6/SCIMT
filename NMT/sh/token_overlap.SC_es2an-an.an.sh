python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.an \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an.an/SC_es2an.an \
    --data2 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.es.SC_es2an \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an.an/SC_es2an.an \
    --is_parallel \
    --out sh/token_overlap.SC_es2an.an.parallel.json

python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.an \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an.an/SC_es2an.an \
    --data2 /home/hatch5o6/nobackup/archive/data/LRRomance/es-en/CCMatrix/fixed/cleaned/tgt.100k.SC_es2an.txt \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/SC_es2an.an/SC_es2an.an \
    --out sh/token_overlap.SC_es2an.an.json
