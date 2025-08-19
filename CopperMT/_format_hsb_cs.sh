echo "Created by /home/hatch5o6/Cognate/code/CopperMT/format_hsb_cs.sh" > /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs/0/notes.txt

mkdir /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs
mkdir /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs/0
python format_data.py \
    --data /home/hatch5o6/nobackup/archive/data/cs-hsb/WMT-CCMatrix-Overlap/fastalign/CogNet-v2.0-EtymBD-word_list-hsb.cs.s=1500.tsv \
    --src hsb \
    --tgt cs \
    --out_dir /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs/0 \
    --prefix train

python format_data.py \
    --data /home/hatch5o6/nobackup/archive/data/CogNet/hsb/CogNet-v2.0-EtymDB.hsb.ces.test-s=1420.tsv \
    --src hsb \
    --tgt cs \
    --out_dir /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs/0 \
    --prefix test


python format_data.py \
    --data /home/hatch5o6/nobackup/archive/data/CogNet/hsb/CogNet-v2.0-EtymDB.hsb.ces.val-s=1420.tsv \
    --src hsb \
    --tgt cs \
    --out_dir /home/hatch5o6/nobackup/archive/CopperMT/inputs/split_data/hsb_cs/0 \
    --prefix fine_tune