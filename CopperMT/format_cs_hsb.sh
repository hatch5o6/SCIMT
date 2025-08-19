SEED=0
SRC=cs
TGT=hsb
COPPER_DIR=/home/hatch5o6/nobackup/archive/CopperMT/${SRC}_${TGT}
if [ -d $COPPER_DIR ]; then
    echo "deleting ${COPPER_DIR}"
    rm -r $COPPER_DIR
fi
mkdir $COPPER_DIR
mkdir ${COPPER_DIR}/inputs
mkdir ${COPPER_DIR}/inputs/split_data
OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/${SRC}_${TGT}/inputs/split_data/${SRC}_${TGT}
python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/cs.train-s=1420.txt,/home/hatch5o6/nobackup/archive/data/cs-hsb/WMT-CCMatrix-Overlap/fastalign/word_list.hsb-ce.cognates.0.5.parallel-cs.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/hsb.train-s=1420.txt,/home/hatch5o6/nobackup/archive/data/cs-hsb/WMT-CCMatrix-Overlap/fastalign/word_list.hsb-ce.cognates.0.5.parallel-hsb.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix train \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/cs.test-s=1420.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/hsb.test-s=1420.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix test \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/cs.val-s=1420.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/CogNet/ces-hsb/combined_with_etymdb/split/hsb.val-s=1420.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix fine_tune \
    --seed $SEED

echo "Created by /home/hatch5o6/Cognate/code/CopperMT/format_cs_hsb.sh" > $OUT_DIR/${SEED}/notes.txt