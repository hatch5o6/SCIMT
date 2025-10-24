LANGS=bho-hi,bn-as,cs-hsb,djk-en,en-djk,es-an,fr-mfe,hi-bho,lua-bem

python Pipeline/compile_hyperparam_search_results.py \
    --langs $LANGS \
    --tag CUR > /home/hatch5o6/Cognate/code/Pipeline/sh/compile_hyperparam_search_results.out