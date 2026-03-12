
echo "Removing rnn_hyperparams"
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/*.txt
rm /home/hatch5o6/Cognate/code/Pipeline/rnn_hyperparams/manifest.json

echo "Removing sbatch files"
rm /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_oc_abl/*

echo "Removing cfgs"
rm -r /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH_OC_ABL

echo "Removing slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/hyper_param_search_outputs_oc_abl/*

echo "Removing smt slurm outputs"
rm /home/hatch5o6/Cognate/code/Pipeline/slurm_outputs/SC_smt_oc_abl/*

echo "Removing parameters"
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*FR-OC*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*ESX-ANX*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*FRX-MFX*
rm /home/hatch5o6/Cognate/code/Pipeline/parameters/*FRY-MFY*

echo "Removing CoppertMT lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/CopperMT/FR-OC*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/ESX-*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/FRX-*
rm -r /home/hatch5o6/nobackup/archive/CopperMT/FRY-*

echo "Removing COGNATE_TRAIN lang subdirs"
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fr-oc*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/esx-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/frx-*
rm -r /home/hatch5o6/nobackup/archive/data/COGNATE_TRAIN/fry-*

python Pipeline/make_hyperparam_search_space.py \
    --new_cfg_dir /home/hatch5o6/Cognate/code/Pipeline/cfg/SC-HYPERPARAM_SEARCH_OC_ABL \
    --sbatch_dir /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_oc_abl \
    --cfgs /home/hatch5o6/Cognate/code/Pipeline/cfg/SC/fr-oc.cfg,/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_ablations/esx-anx.cfg,/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_ablations/frx-mfx.cfg,/home/hatch5o6/Cognate/code/Pipeline/cfg/SC_ablations/fry-mfy.cfg\
    --tag oc_abl

# echo "CREATED SEARCH SPACE BUT DID NOT RUN"
# exit

echo "RNN SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/hyper_param_search_oc_abl/*
do
    echo "    $f"
    sbatch $f
done

echo ""
echo "SMT SBATCH:-"
for f in /home/hatch5o6/Cognate/code/Pipeline/sbatch/smt_oc_abl/*
do
    echo "    $f"
    sbatch $f
done