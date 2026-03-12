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
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*BN-AS*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*CS-HSB*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*ES-AN*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*FR-MFE*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*BHO-HI*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*DJK-EN*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*EN-DJK*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*EWE-FON*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*FON-EWE*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*HI-BHO*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*LUA-BEM*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*AR-AEB*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*AR-APC*

echo "Removing CoppertMT lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/CopperMT/BN_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/CS_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/ES_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/FR_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/BHO_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/DJK_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/EN_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/EWE_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/FON_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/HI_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/LUA_*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/AR_*

echo "Removing COGNATE_TRAIN lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/bn-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/cs-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/es-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fr-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/bho-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/djk-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/en-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/ewe-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fon-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/hi-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/lua-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/ar-*

# echo "ONLY DELETED FILES AND DIRS"
# exit

python Pipeline/make_hyperparam_search_space.py \
    --cfgs Pipeline/cfg/SC/fr-mfe.cfg,Pipeline/cfg/SC/es-an.cfg,Pipeline/cfg/SC/bn-as.cfg,Pipeline/cfg/SC/bho-hi.cfg,Pipeline/cfg/SC/djk-en.cfg,Pipeline/cfg/SC/ewe-fon.cfg,Pipeline/cfg/SC/fon-ewe.cfg,Pipeline/cfg/SC/hi-bho.cfg,Pipeline/cfg/SC/lua-bem.cfg,Pipeline/cfg/SC/en-djk.ATT.cfg,Pipeline/cfg/SC/ar-aeb.cfg,Pipeline/cfg/SC/ar-apc.cfg

# echo "CREATED SEARCH SPACE BUT DID NOT RUN"
# exit

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