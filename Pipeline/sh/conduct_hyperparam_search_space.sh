# Search with these langs, which have real test data and a variety of train sizes:
#   fr-mfe 	7.3k train segs
#   es-an	56k train segs
#   cs-hsb	916 train segs
#   bn-as	246k train segs

echo "Removing rnn_hyperparams"
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/*.txt
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/manifest.json

echo "Removing sbatch files"
rm /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search/*

echo "Removing cfgs"
rm -r /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH

echo "Removing slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs/*
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_/*

echo "Removing smt slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt/*

echo "Removing parameters"
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*bn-as*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*cs-hsb*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*es-an*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*fr-mfe*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*bho-hi*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*djk-en*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*ewe-fon*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*fon-ewe*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*hi-bho*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*lua-bem*

echo "Removing CoppertMT lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/CopperMT/bn_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/cs_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/es_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/fr_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/bho_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/djk_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/ewe_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/fon_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/hi_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/lua_*

echo "Removing COGNATE_TRAIN lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/bn-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/cs-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/es-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fr-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/bho-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/djk-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/ewe-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fon-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/hi-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/lua-*


python Pipeline/make_hyperparam_search_space.py \
    --cfgs Pipeline/cfg/SC/fr-mfe.cfg,Pipeline/cfg/SC/es-an.cfg,Pipeline/cfg/SC/cs-hsb.cfg,Pipeline/cfg/SC/bn-as.cfg,Pipeline/cfg/SC/bho-hi.cfg,Pipeline/cfg/SC/djk-en.cfg,Pipeline/cfg/SC/ewe-fon.cfg,Pipeline/cfg/SC/fon-ewe.cfg,Pipeline/cfg/SC/hi-bho.cfg,Pipeline/cfg/SC/lua-bem.cfg

echo "RNN SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search/*
do
    echo "    $f"
    sbatch $f
done

echo ""
echo "SMT SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/smt/*
do
    echo "    $f"
    sbatch $f
done