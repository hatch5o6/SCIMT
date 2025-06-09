python get_cognate_diff.py \
    -c1 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/fastalign/word_list.fra-mfe.cognates.0.5.txt \
    -c2 /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/no_grouping_fastalign/word_list.fra-mfe.cognates.0.5.txt \
    --out1_insert curr-no_grouping \
    --out2_insert no_grouping-curr

python get_cognate_diff.py \
    -c1 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/fastalign/word_list.an-es.cognates.0.5.txt \
    -c2 /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/no_grouping_fastalign/word_list.an-es.cognates.0.5.txt \
    --out1_insert curr-no_grouping \
    --out2_insert no_grouping-curr