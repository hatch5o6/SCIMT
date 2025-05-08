FILE=/home/hatch5o6/nobackup/archive/data/EMNLP2021/fastalign/train.hsb-de

./../fast_align/build/fast_align \
    -i ${FILE}.txt \
    -d -o -v > ${FILE}.forward.align

./../fast_align/build/fast_align \
    -i ${FILE}.txt \
    -d -o -v -r > ${FILE}.reverse.align

./../fast_align/build/atools \
    -i ${FILE}.forward.align \
    -j ${FILE}.reverse.align -c grow-diag-final-and > ${FILE}.sym.align
