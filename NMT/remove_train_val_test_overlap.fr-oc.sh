python NMT/remove_train_val_test_overlap.py \
    --src fr \
    --tgt oc \
    --train /home/hatch5o6/nobackup/archive/data/NLLB/fr_oc/cleaned/src.txt,/home/hatch5o6/nobackup/archive/data/NLLB/fr_oc/cleaned/tgt.txt \
    --val /home/hatch5o6/nobackup/archive/data/FLORES+/dev/fra_Latn_stan1290.dev,/home/hatch5o6/nobackup/archive/data/FLORES+/dev/oci_Latn_occi1239.dev \
    --test /home/hatch5o6/nobackup/archive/data/FLORES+/devtest/fra_Latn_stan1290.devtest,/home/hatch5o6/nobackup/archive/data/FLORES+/devtest/oci_Latn_occi1239.devtest \
    --out_dir /home/hatch5o6/nobackup/archive/data/CharLOTTE_occitan/fr-oc
