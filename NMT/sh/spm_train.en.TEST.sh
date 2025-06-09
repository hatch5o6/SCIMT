# Training size is 57188 = 28594 * 2 (for es + an)
# I'm keeping the training size the same as the es.an and SC_es2an.an models
python spm_train.py \
    --data "{'/home/hatch5o6/nobackup/archive/data/LRRomance/es-en/CCMatrix/fixed/cleaned/src.100k.txt': 1.0}" \
    --training_data_size 57188 \
    --save_dir /home/hatch5o6/nobackup/archive/CognateMT/spm_models/TEST/en \
    --spm_model_name en \
    --user_defined_symbols "<pad>,<en>,<es>,<an>"
