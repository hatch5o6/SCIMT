python NMT/prepare_human_eval_data.test.py \
    --test "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.en.txt" \
    --hyp1 "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.es-en>>an-en_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt" \
    --hyp2 "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.SC_es2an-en>>an-en_TRIAL_s=1000/predictions/epoch=5-step=3768-val_loss=2.1628.ckpt/test_predictions.txt" \
    --mt_eval "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/mt_eval/just_ref.tsv"