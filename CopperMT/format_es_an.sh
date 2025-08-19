SEED=0
SRC=es
TGT=an
OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/${SRC}_${TGT}
python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/fastalign/word_list.an-es.cognates.0.5.parallel-es.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined/fastalign/word_list.an-es.cognates.0.5.parallel-an.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix train \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/CogNet/spa-arg/combined_with_etymdb/split_UT/es.test-s=1420.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/CogNet/spa-arg/combined_with_etymdb/split_UT/an.test-s=1420.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix test \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/CogNet/spa-arg/combined_with_etymdb/split_UT/es.val-s=1420.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/CogNet/spa-arg/combined_with_etymdb/split_UT/an.val-s=1420.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix fine_tune \
    --seed $SEED

echo "Created by /home/hatch5o6/Cognate/code/CopperMT/format_es_an.sh" > $OUT_DIR/${SEED}/notes.txt