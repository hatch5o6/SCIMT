
python train_hf_tokenizer.py \
    --data "{'/home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/concat.mfe-fra.mfe': .5, '/home/hatch5o6/nobackup/archive/data/CCMatrix_fr_en/fixed/stitched/cleaned/tgt.50k.SC_NGfr2NGmfe.txt': .5}" \
    --training_data_size 50000 \
    --save_dir /home/hatch5o6/nobackup/archive/CognateMT/hf_tokenizers/SC_fr2mfe.mfe \
    --model_name SC_fr2mfe.mfe \
    --special_tokens "<unk>,<s>,</s>,<pad>,<en>,<fr>,<mfe>"
