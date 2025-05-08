SRC=an
TGT=es
DATA=/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined
SRC_F=${DATA}/train/train.${SRC}
TGT_F=${DATA}/train/train.${TGT}
mkdir ${DATA}/fastalign
SRC_TGT_F=${DATA}/fastalign/${SRC}-${TGT}
WORD_LIST_OUT=${DATA}/fastalign/word_list.${TGT}2${SRC}.txt

# prepare
python prepare_for_fastalign.py \
    --src $SRC_F \
    --tgt $TGT_F \
    --out ${SRC_TGT_F}.txt

# fast_align
./../../fast_align/build/fast_align \
    -i ${SRC_TGT_F}.txt \
    -d -o -v -r > ${SRC_TGT_F}.reverse.align

# make word alignments
python make_word_alignments.py \
    --alignments ${SRC_TGT_F}.reverse.align \
    --sent_pairs ${SRC_TGT_F}.txt \
    --out $WORD_LIST_OUT \
    --VERBOSE \
    --STOP 10

    