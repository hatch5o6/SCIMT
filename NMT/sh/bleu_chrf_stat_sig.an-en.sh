echo "######################### an-en #########################"

REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.en.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.es-en>>an-en_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.SC_es2an-en>>an-en_TRIAL_s=1000/predictions/epoch=5-step=3768-val_loss=2.1628.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar




echo ""
echo ""
echo ""
echo "######################### an-en.REVERSE #########################"
REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.an.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.es-en>>an-en.REVERSE_TRIAL_s=1000/predictions/epoch=8-step=5652-val_loss=3.2223.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/an-en/FINETUNE.SC_es2an-en>>an-en.REVERSE_TRIAL_s=1000/predictions/epoch=6-step=4396-val_loss=3.0011.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar