echo "CHECK DEV"
python Pipeline/diff_check.py\
    -f1  /home/hatch5o6/nobackup/archive/data/flores200_dataset/dev/spa_Latn.dev \
    -f2  /home/hatch5o6/nobackup/archive/data/LRRomance/PILAR/FLORES+/dev/dev.spa_Latn \
    -o Pipeline/sh_outputs/get_spa_flores200_vs_pilar_flores+_diffs.dev.out

echo ""
echo ""
echo "CHECKING DEVTEST"
python Pipeline/diff_check.py\
    -f1 /home/hatch5o6/nobackup/archive/data/flores200_dataset/devtest/spa_Latn.devtest \
    -f2 /home/hatch5o6/nobackup/archive/data/LRRomance/PILAR/FLORES+/devtest/devtest.spa_Latn \
    -o Pipeline/sh_outputs/get_spa_flores200_vs_pilar_flores+_diffs.devtest.out