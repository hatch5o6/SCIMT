python get_overlap.py \
    --lang1 /home/hatch5o6/nobackup/archive/data/NLLB/ar_en/cleaned/src.txt \
    --com1 /home/hatch5o6/nobackup/archive/data/NLLB/ar_en/cleaned/tgt.txt \
    --lang2 /home/hatch5o6/nobackup/archive/data/NLLB/mt_en/cleaned/tgt.txt \
    --com2 /home/hatch5o6/nobackup/archive/data/NLLB/mt_en/cleaned/src.txt \
    --lang1_out /home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/ar.txt \
    --lang2_out /home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/mt.txt

echo "Created by /home/hatch5o6/Cognate/code/word_alignments/get_overlap.mt-ar.sh" > /home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/notes
date >> /home/hatch5o6/nobackup/archive/data/NLLB/mt_ar_overlap/notes