python Pipeline/sc_inference_confidence_dist.py \
    -i /home/hatch5o6/nobackup/archive/CopperMT/en_djk_RNN-121_S-0/workspace/reference_models/bilingual/rnn_en-djk/0/results/inference_selected_checkpoint_en_djk.djk/generate-test.txt \
    -o /home/hatch5o6/Cognate/code/Pipeline/en-djk_hyps

python Pipeline/sc_inference_confidence_dist.py \
    -i /home/hatch5o6/nobackup/archive/CopperMT/es_an_RNN-285_S-0/workspace/reference_models/bilingual/rnn_es-an/0/results/inference_selected_checkpoint_es_an.an/generate-test.txt\
    -o /home/hatch5o6/Cognate/code/Pipeline/es-an_hyp

python Pipeline/sc_inference_confidence_dist.py \
    -i /home/hatch5o6/nobackup/archive/CopperMT/fr_mfe_RNN-66_S-0/workspace/reference_models/bilingual/rnn_fr-mfe/0/results/inference_selected_checkpoint_fr_mfe.mfe/generate-test.txt \
    -o /home/hatch5o6/Cognate/code/Pipeline/fr-mfe_hyp
