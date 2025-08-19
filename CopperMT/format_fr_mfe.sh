SEED=0
SRC=fr
TGT=mfe
OUT_DIR=/home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/${SRC}_${TGT}
python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/fastalign/word_list.fra-mfe.cognates.0.5.parallel-fra.txt \
    --tgt_data /home/hatch5o6/nobackup/archive/data/Kreyol-MT/mfe-fra/fastalign/word_list.fra-mfe.cognates.0.5.parallel-mfe.txt \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix train \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/EtymDB/fr_mfe/train/orig.fr_mfe.test-s=1420.fr \
    --tgt_data /home/hatch5o6/nobackup/archive/data/EtymDB/fr_mfe/train/orig.fr_mfe.test-s=1420.mfe \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix test \
    --seed $SEED

python format_data.py \
    --src_data /home/hatch5o6/nobackup/archive/data/EtymDB/fr_mfe/train/orig.fr_mfe.val-s=1420.fr \
    --tgt_data /home/hatch5o6/nobackup/archive/data/EtymDB/fr_mfe/train/orig.fr_mfe.val-s=1420.mfe \
    --src $SRC \
    --tgt $TGT \
    --out_dir $OUT_DIR \
    --prefix fine_tune \
    --seed $SEED

echo "Created by /home/hatch5o6/Cognate/code/CopperMT/format_fr_mfe.sh" > $OUT_DIR/${SEED}/notes.txt