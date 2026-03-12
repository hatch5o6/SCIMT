python NMT/remove_train_val_test_overlap.py \
    --src en \
    --tgt oc \
    --train /home/hatch5o6/nobackup/archive/data/NLLB/en_oc/cleaned/src.txt,/home/hatch5o6/nobackup/archive/data/NLLB/en_oc/cleaned/tgt.txt \
    --val /home/hatch5o6/nobackup/archive/data/FLORES+/dev/eng_Latn_stan1293.dev,/home/hatch5o6/nobackup/archive/data/FLORES+/dev/oci_Latn_occi1239.dev \
    --test /home/hatch5o6/nobackup/archive/data/FLORES+/devtest/eng_Latn_stan1293.devtest,/home/hatch5o6/nobackup/archive/data/FLORES+/devtest/oci_Latn_occi1239.devtest \
    --out_dir /home/hatch5o6/nobackup/archive/data/CharLOTTE_occitan/en-oc