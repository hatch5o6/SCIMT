# Search with these langs, which have real test data and a variety of train sizes:
#   fr-mfe 	7.3k train segs
#   es-an	56k train segs
#   cs-hsb	916 train segs
#   bn-as	246k train segs

echo "Removing rnn_hyperparams"
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/*.txt
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/manifest.json

echo "Removing sbatch files"
rm /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_NG-False/*

echo "Removing cfgs"
rm -r /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH_NG-False

echo "Removing slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_NG-False/*

echo "Removing smt slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt_NG-False/*

echo "Removing parameters"
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*bn-as*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*cs-hsb*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*es-an*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*fr-mfe*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*bho-hi*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*djk-en*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*ewe-fon*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*fon-ewe*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*hi-bho*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters_NG-False/*lua-bem*

echo "Removing CoppertMT lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/bn_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/cs_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/es_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/fr_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/bho_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/djk_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/ewe_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/fon_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/hi_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT_NG-False/lua_*

echo "Removing COGNATE_TRAIN lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/bn-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/cs-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/es-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/fr-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/bho-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/djk-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/ewe-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/fon-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/hi-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN_NG-False/lua-*


python Pipeline/make_hyperparam_search_space.py \
    --cfgs Pipeline/cfg/SC_NG=False/fr-mfe.cfg,Pipeline/cfg/SC_NG=False/es-an.cfg,Pipeline/cfg/SC_NG=False/cs-hsb.cfg,Pipeline/cfg/SC_NG=False/bn-as.cfg,Pipeline/cfg/SC_NG=False/bho-hi.cfg,Pipeline/cfg/SC_NG=False/djk-en.cfg,Pipeline/cfg/SC_NG=False/ewe-fon.cfg,Pipeline/cfg/SC_NG=False/fon-ewe.cfg,Pipeline/cfg/SC_NG=False/hi-bho.cfg,Pipeline/cfg/SC_NG=False/lua-bem.cfg \
    --new_cfg_dir /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH_NG-False \
    --sbatch_dir /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_NG-False \
    --tag "NG-False"
# python Pipeline/make_hyperparam_search_space.py \
#     --cfgs Pipeline/cfg/SC_NG=False/fr-mfe.cfg \
#     --new_cfg_dir /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH_NG-False \
#     --sbatch_dir /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_NG-False \
#     --tag "NG-False"

echo "RNN SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_NG-False/*
do
    echo "    $f"
    sbatch $f
done

echo ""
echo "SMT SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/smt_NG-False/*
do
    echo "    $f"
    sbatch $f
done