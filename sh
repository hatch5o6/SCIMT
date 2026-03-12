######################### anx-enx #########################
sacrebleu "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/anx-enx/test.enx.txt" -i "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt" "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt" -m bleu chrf -w 4
[
    {
        "system": "Baseline: /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt",
        "BLEU": {
            "score": 18.152839790197053,
            "p_value": null,
            "mean": null,
            "ci": null
        },
        "chrF2": {
            "score": 47.83625108844321,
            "p_value": null,
            "mean": null,
            "ci": null
        }
    },
    {
        "system": "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt",
        "BLEU": {
            "score": 19.16595795850805,
            "p_value": 0.0034996500349965005,
            "mean": null,
            "ci": null
        },
        "chrF2": {
            "score": 49.42447057947058,
            "p_value": 9.999000099990002e-05,
            "mean": null,
            "ci": null
        }
    }
]
######################### anx-enx again #########################
sacrebleu "/home/hatch5o6/nobackup/archive/data/CharLOTTE_data/an-en/test.en.txt" -i "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt" "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt" -m bleu chrf -w 4
[
    {
        "system": "Baseline: /home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.esx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=7-step=5024-val_loss=2.3692.ckpt/test_predictions.txt",
        "BLEU": {
            "score": 18.152839790197053,
            "p_value": null,
            "mean": null,
            "ci": null
        },
        "chrF2": {
            "score": 47.83625108844321,
            "p_value": null,
            "mean": null,
            "ci": null
        }
    },
    {
        "system": "/home/hatch5o6/nobackup/archive/CognateMT/PredictCognates/anx-enx/FINETUNE.SC_esx2anx-enx>>anx-enx_TRIAL_s=1000/predictions/epoch=5-step=3454-val_loss=2.1988.ckpt/test_predictions.txt",
        "BLEU": {
            "score": 19.16595795850805,
            "p_value": 0.0034996500349965005,
            "mean": null,
            "ci": null
        },
        "chrF2": {
            "score": 49.42447057947058,
            "p_value": 9.999000099990002e-05,
            "mean": null,
            "ci": null
        }
    }
]
