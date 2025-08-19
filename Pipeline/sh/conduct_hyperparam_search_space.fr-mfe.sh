# Search with these langs, which have real test data and a variety of train sizes:
#   fr-mfe 	7.3k train segs
#   es-an	56k train segs
#   cs-hsb	916 train segs
#   bn-as	246k train segs

echo "Removing rnn_hyperparams"
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/*.txt
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/manifest.json
echo "Removing sbatch files"
rm /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search/fr-*
echo "Removing cfgs"
rm /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH/fr-*
# echo "Removing slurm outputs"
# rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs/*

echo "Removing CoppertMT lang subdirs"
# rm -r /home/hatch5o6/nobackup/archive/CopperMT/bn_*
# rm -r /home/hatch5o6/nobackup/archive/CopperMT/cs_*
# rm -r /home/hatch5o6/nobackup/archive/CopperMT/es_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/fr_*

echo "Removing smt slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt/*fr-mfe*

echo "Removing COGNATE_TRAIN lang subdirs"
# rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/bn-*
# rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/cs-*
# rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/es-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fr-*


python Pipeline/make_hyperparam_search_space.py \
    --cfgs Pipeline/cfg/SC/fr-mfe.cfg

echo "SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search/*
do
    echo "    $f"
    sbatch $f
done

echo /home/hatch5o6/Cognate/code/Pipeline/sbatch/smt/fr-mfe.smt.cfg.sh
sbatch /home/hatch5o6/Cognate/code/Pipeline/sbatch/smt/fr-mfe.smt.cfg.sh