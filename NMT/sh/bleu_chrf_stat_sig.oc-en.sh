echo "######################### oc-en #########################"

REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_occitan/en-oc/test.en.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/oc-en/FINETUNE.fr-en>>oc-en_TRIAL_s=1000/predictions/epoch=6-step=21006-val_loss=1.7334.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/oc-en/FINETUNE.SC_fr2oc-en>>oc-en_TRIAL_s=1000/predictions/epoch=6-step=21006-val_loss=1.6603.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar




echo ""
echo ""
echo ""
echo "######################### oc-en.REVERSE #########################"
REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_occitan/en-oc/test.oc.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/oc-en/FINETUNE.fr-en>>oc-en.REVERSE_TRIAL_s=1000/predictions/epoch=4-step=15004-val_loss=2.2843.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/oc-en/FINETUNE.SC_fr2oc-en>>oc-en.REVERSE_TRIAL_s=1000/predictions/epoch=6-step=21006-val_loss=2.3594.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar