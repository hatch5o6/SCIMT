echo "######################### mfe-en #########################"

REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/mfe-en/test.en.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en/FINETUNE.fr-en>>mfe-en_TRIAL_s=1000/predictions/epoch=12-step=7487-val_loss=2.7501.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en/FINETUNE.SC_fr2mfe-en>>mfe-en_TRIAL_s=1000/predictions/epoch=6-step=4192-val_loss=2.3282.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar




echo ""
echo ""
echo ""
echo "######################### mfe-en.REVERSE #########################"
REF="/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/mfe-en/test.mfe.txt"
BAS="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en/FINETUNE.fr-en>>mfe-en.REVERSE_TRIAL_s=1000/predictions/epoch=12-step=7786-val_loss=3.5993.ckpt/test_predictions.txt"
HYP="/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/mfe-en/FINETUNE.SC_fr2mfe-en>>mfe-en.REVERSE_TRIAL_s=1000/predictions/epoch=9-step=5989-val_loss=3.1561.ckpt/test_predictions.txt"

echo "sacrebleu \"$REF\" -i \"$BAS\" \"$HYP\" -m bleu chrf -w 4"

sacrebleu "$REF" -i "$BAS" "$HYP" -m bleu chrf -w 4 --paired-ar