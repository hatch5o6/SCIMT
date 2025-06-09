python evaluate.py \
    --ref /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/inputs/split_data/cs_hsb/0/test_cs_hsb.cs \
    --hyp /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/bilingual/rnn_cs-hsb/0/results/results_checkpoint_best_cs_hsb.hsb/generate-test.hyps.txt \
    --out /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/bilingual/rnn_cs-hsb/0/results/results_checkpoint_best_cs_hsb.hsb/generate-test.hyps.scores.tested_against_cs.txt

python evaluate.py \
    --ref /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/inputs/split_data/cs_hsb/0/test_cs_hsb.cs \
    --hyp /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/statistical/0/cs_hsb/out/test_cs_hsb_translated_10.hsb \
    --out /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/statistical/0/cs_hsb/out/test_cs_hsb_translated_10.hsb.scores.tested_against_cs


python evaluate.py \
    --ref /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/inputs/split_data/cs_hsb/0/test_cs_hsb.hsb \
    --hyp /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/bilingual/rnn_cs-hsb/0/results/results_checkpoint_best_cs_hsb.hsb/generate-test.hyps.txt \
    --out /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/bilingual/rnn_cs-hsb/0/results/results_checkpoint_best_cs_hsb.hsb/generate-test.hyps.scores.tested_against_hsb.txt

python evaluate.py \
    --ref /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/inputs/split_data/cs_hsb/0/test_cs_hsb.hsb \
    --hyp /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/statistical/0/cs_hsb/out/test_cs_hsb_translated_10.hsb \
    --out /home/hatch5o6/nobackup/archive/CopperMT/cs_hsb/workspace/reference_models/statistical/0/cs_hsb/out/test_cs_hsb_translated_10.hsb.scores.tested_against_hsb