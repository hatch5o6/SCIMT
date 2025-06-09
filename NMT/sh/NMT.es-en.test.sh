sbatch 'sbatch/test_FINETUNE.es.en->an.en.sh'

sbatch sbatch/test_NMT.es-an.en.1M.sh
sbatch sbatch/test_NMT.es-an.en.500k.sh 
sbatch sbatch/test_NMT.es-an.en.sh
sbatch sbatch/test_NMT.es-an.en.upsample.sh 

sbatch sbatch/test_NMT.SC_es2an-an.en.1M.sh
sbatch sbatch/test_NMT.SC_es2an-an.en.500k.sh
sbatch sbatch/test_NMT.SC_es2an-an.en.sh
sbatch sbatch/test_NMT.SC_es2an-an.en.upsample.sh

sbatch sbatch/test_PRETRAIN.es.en.10M.sh

