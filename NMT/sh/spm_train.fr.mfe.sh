
python spm_train.py \
    --data "{'/home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe': .5, '/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.50k.txt': .5}" \
    --training_data_size 50000 \
    --save_dir /home/hatch5o6/nobackup/archive/CognateMT/spm_models/fr.mfe \
    --spm_model_name fr.mfe \
    --user_defined_symbols "<pad>,<en>,<fr>,<mfe>"
