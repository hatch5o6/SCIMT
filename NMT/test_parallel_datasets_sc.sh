python test_parallel_datasets_sc.py \
    --data_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/PLAIN/mfe-en/train.csv \
    --sc_data_csv /home/hatch5o6/Cognate/code/NMT/augmented_data/SC/SC_fr2mfe-en/train.csv \
    --sc_model_id FR-MFE-RNN-0 \
    --pipeline test_ordered,test_shuffled