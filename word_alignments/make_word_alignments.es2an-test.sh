SRC=an
TGT=es
DATA=/home/hatch5o6/nobackup/archive/data/LRRomance/es-an/Combined
SRC_F=${DATA}/train/train.${SRC}
TGT_F=${DATA}/train/train.${TGT}
mkdir ${DATA}/fastalign
SRC_TGT_F=${DATA}/fastalign/${SRC}-${TGT}
WORD_LIST_OUT=${DATA}/fastalign/word_list.${TGT}2${SRC}.TEST.txt

# # prepare
# python prepare_for_fastalign.py \
#     --src $SRC_F \
#     --tgt $TGT_F \
#     --out ${SRC_TGT_F}.txt

# # fast_align
# ./../../fast_align/build/fast_align \
#     -i ${SRC_TGT_F}.txt \
#     -d -o -v > ${SRC_TGT_F}.forward.align

# ./../../fast_align/build/fast_align \
#     -i ${SRC_TGT_F}.txt \
#     -d -o -v -r > ${SRC_TGT_F}.reverse.align

# ./../../fast_align/build/atools \
#     -i ${SRC_TGT_F}.forward.align \
#     -j ${SRC_TGT_F}.reverse.align \
#     -c grow-diag-final-and > ${SRC_TGT_F}.sym.align

# make word alignments
python make_word_alignments.py \
    --alignments ${SRC_TGT_F}.reverse.align \
    --sent_pairs ${SRC_TGT_F}.txt \
    --out $WORD_LIST_OUT \
    --VERBOSE \
    --START 8336 \
    --STOP 8340

    