# Training size is 57188 = 28594 * 2 (for es + an)
python spm_train.py \
    --data "{'/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.es': .5, '/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/train/train.an': .5}" \
    --training_data_size 57188 \
    --save_dir /home/hatch5o6/nobackup/archive/CognateMT/spm_models/es.an \
    --spm_model_name es.an \
    --user_defined_symbols "<pad>,<en>,<es>,<an>"
