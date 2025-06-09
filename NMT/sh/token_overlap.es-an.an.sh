python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.an \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es.an/es.an \
    --data2 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.es \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es.an/es.an \
    --is_parallel \
    --out sh/token_overlap.es-en.an.parallel.json

python token_overlap.py \
    --data1 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.an \
    --spm1 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es.an/es.an \
    --data2 /home/hatch5o6/nobackup/archive/data/LRRomance/es-en/CCMatrix/fixed/cleaned/tgt.100k.txt \
    --spm2 /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es.an/es.an \
    --out sh/token_overlap.es-en.an.json
