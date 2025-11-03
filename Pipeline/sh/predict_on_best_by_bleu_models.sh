# Will predict with best models, based on BLEU scores of hyperparameter search models

sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/ar-aeb.102.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/ar-apc.269.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/bho-hi.246.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/bn-as.283.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/cs-hsb.smt.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/djk-en.smt.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/en-djk.121.cfg.sh # this outperforms the smt model by BLEU
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/es-an.285.cfg.sh # this outperforms 284 model by BLEU
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/ewe-fon.120.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/fon-ewe.210.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/fr-mfe.66.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/hi-bho.228.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/lua-bem.249.cfg.sh

# sanity check -- we expect these to fail
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/fr-mfe.100.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/predict/djk-en.6.cfg.sh
