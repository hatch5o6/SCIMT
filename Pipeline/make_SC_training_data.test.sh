# mkdir /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/make_SC_data_test

# python make_SC_training_data.py \
#     --train_csv ../NMT/data/fr-mfe/train.csv \
#     --val_csv ../NMT/data/fr-mfe_dev_test/val.csv \
#     --test_csv ../NMT/data/fr-mfe_dev_test/test.csv \
#     --src_out /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/make_SC_data_test/train.fr \
#     --tgt_out /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/make_SC_data_test/train.mfe \
#     --src fr \
#     --tgt mfe


mkdir /home/hatch5o6/nobackup/archive/data/Kreyol-MT/bren-dan_TEST/make_SC_data_test

python make_SC_training_data.py \
    --train_csv ../NMT/data/bren-dan/train.csv \
    --val_csv ../NMT/data/bren-dan_dev_test/val.csv \
    --test_csv ../NMT/data/bren-dan_dev_test/test.csv \
    --src_out /home/hatch5o6/nobackup/archive/data/Kreyol-MT/bren-dan_TEST/make_SC_data_test/train.bren \
    --tgt_out /home/hatch5o6/nobackup/archive/data/Kreyol-MT/bren-dan_TEST/make_SC_data_test/train.dan \
    --src bren \
    --tgt dan