echo "######################### anx-enx #########################"

REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/anx-enx/test.enx.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar




echo "######################### anx-enx again #########################"

REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.en.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar



