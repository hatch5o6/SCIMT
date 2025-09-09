# Clone CopperMT and add new/updated scripts
From root directory, run these:
```
cd CopperMT
git clone https://github.com/clefourrier/CopperMT.git
cd ../CopperMTfiles
python move_files.py
```

# Pipeline

See *Pipeline/cfg/SC* for the .cfg files for all 10 scenarios of these experiments. They contain the following parameters. If ever not using one of these parameters, as relvant (most should be used), then set it to null. See Pipeline/cfg/SC for details.
- **MODULE_HOME_DIR:** the system path to the *code* folder of this module, depending on where you cloned it on your system, e.g. *~/path/to/Cognate/code*
- **NMT_SRC:** source language in the low-resource (LR) direction we ultimately want to translate. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **NMT_TGT:** target language in the low-resource (LR) direction we ultimately want to translate. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **AUG_SRC:** source language of the high-resource (HR) direction we want to levarage. Should be a high-resource (HR) language closely related to *NMT_SRC*. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **AUG_TGT:** target language of the high-resource direction we want to leverage. Should be **THE SAME AS** *NMT_TGT*. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **SRC:** the source language of the cognate prediction model. This should be the same as *AUG_SRC*. The goal is to use the resulting cognate prediction model to make *AUG_SRC* look more like *NMT_SRC* based on character correspondences.
- **TGT:** the target language of the cognate prediction model. This should be the same as *NMT_SRC*. The goal is to use the resulting cognate prediction model to make *AUG_SRC* look more like *NMT_SRC* based on character correspondences.
- **SEED:** a random seed used in different scripts, such as for randomizing data order
- **PARALLEL_TRAIN/VAL/TEST** Parallel train / val / test data .csv files. These are the parallel data used to train NMT models, and from which congates will be extracted to train the cognate prediction model.
- **APPLY_TO** list (comma-delimited, no space) of more data .csv files to apply the cognate prediction model to. Not used by *train_SC.sh* but by *pred_SC.sh*.
- **NO_GROUPING** Keep this set to True. Not sure I'll actually experiment with this. It's used when extracting the cognate list from the Fast Align results. Basically, if False, then "grouping" is applied. Don't worry about it. Ask Brendan if you really want to know.
- **SC_MODEL_TYPE** 'RNN' or 'SMT'. Determines what kind of model will be trained to predict cognates.
- **COGNATE_TRAIN** Directory where Fast Align results and cognate word lists are written. The final training data, however, will be created in *COPPERMT_DATA_DIR*. Don't ask why. It's inefficient copying of data in multiple places and I don't want to fix it at this point.
- **COGNATE_THRESH** the normalized edit distance threshold to determine cognates. Parallel translation data is given to FastAlign which creats word pairs. Words pairs where the normalized edit distance is less than or equal to *COGNATE_THRESH* are considered cognates.
- **COPPERMT_DATA_DIR** Directory where the cognate training data, model checkpoints, and predictions for each scenario will be saved. Each scenario will have its own subdirectory in this directory called *{SRC}_{TGT}_{SMT_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S={SEED}*, *e.g.*, *fr_mfe_RNN-0_S-0*.
- **COPPERMT_DIR** The directory where the CopperMT repo was cloned, *e.g*, */home/hatch5o6/Cognate/code/CopperMT/CopperMT*.
- **PARAMETERS_DIR** A folder to save the CopperMT parameters files
- **RNN_HYPERPARAMS** A folder containing RNN hyperparameter files (each containing a hyperparameter set) and a *manifest.json* file mapping an id to each hyperparameter set (file) (RNNs only).
- **RNN_HYPERPARAMS_ID** The RNN hyperparameter set (see *RNN_HYPERPARAMS*) to use to train an RNN model (RNNs only).
- **BEAM** The number of beams used in beam-search decoding (RNNs only).
- **NBEST** The number of hypotheses to generate. This should just be 1 (Not sure why it's even parameterized). (RNNs only).
- **REVERSE_SRC_TGT_COGNATES** Reverses the *SRC* and *TGT* (cognate pair direction). Should be False. I'm not sure this should ever be True.
- **SC_MODEL_ID** an ID given to the resulting cognate prediction model. This ID is used in other pipelines. Not used by train_SC.sh, but is used by pred_SC.sh to label the resulting noramlized high-resource (norm HR) file (the file that has replaced all words in the HR file with the respective predicted cognate).
- **ADDITIONAL_TRAIN_COGNATES_SRC/TGT** Parallel cognate files if wanting to add data from other sources, such as CogNet or EtymDB, to the training data. If not using, set to 'null'
- **VAL/TEST_COGNATES_SRC/TGT** Set these to the validation/test src/tgt files. If not passed, you should set *COGNATE_TRAIN/VAL/TEST_RATIO* to make train / val / test splits instead. If not using, set to 'null'. Should use either this or *COGNATE_TRAIN/VAL/TEST_RATIO*.
- **COGNATE_TRAIN/VAL/TEST_RATIO** If not passing *VAL/TEST_COGNATES_SRC/TGT*, then these are the train / val / test ratios for splitting the cognate data. The three should add to 1. If not using, set to 'null'. Should use either this or *VAL/TEST_COGNATES_SRC/TGT*.


## Pipeline/train_SC.sh
**Pipeline/train_SC.sh** trains the character correspondence (SC) models.
We call it SC, which stands for "sound correspondence", but more accurately, what we're detecting are character correspondences.

**Pipeline/train_SC.sh** is run from /Cognate/code, and takes a single positional argument, one of the *.cfg* config files described above, e.g.:
```
bash Pipeline/train_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC/fr-mfe.cfg
```

### 1) ARGUMENTS
It uses these parameters from the *.cfg* file: 
- MODULE_HOME_DIR
- SRC
- TGT
- PARALLEL_TRAIN
- PARALLEL_VAL
- PARALLEL_TEST
- COGNATE_TRAIN
- NO_GROUPING
- SC_MODEL_TYPE
- SEED
- SC_MODEL_ID
- COGNATE_THRESH
- COPPERMT_DATA_DIR
- COPPERMT_DIR
- PARAMETERS_DIR
- RNN_HYPERPARAMS
- RNN_HYPERPARAMS_ID
- BEAM
- NBEST
- REVERSE_SRC_TGT_COGNATES
- ADDITIONAL_TRAIN_COGNATES_SRC
- ADDITIONAL_TRAIN_COGNATES_TGT
- VAL_COGNATES_SRC
- VAL_COGNATES_TGT
- TEST_COGNATES_SRC
- TEST_COGNATES_TGT
- COGNATE_TRAIN_RATIO
- COGNATE_TEST_RATIO
- COGNATE_VAL_RATIO

### 2) GET COGNATES FROM PARALLEL DATA
#### 2.1 Clear and remake COGNATE_TRAIN dir
We add *SC_MODEL_TYPE*, *RNN_HYPERPARAMS_ID*, and *SEED* to *COGNATE_TRAIN* directory name. From hereon, when *COGNATE_TRAIN* is mentioned, it will refer to *{COGNATE_TRAIN}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}*. 

If it exists, COGNATE_TRAIN is destroyed and recreated. The COGNATE_TRAIN directory is where the cognate detection parallel data and results get written and saved. It has two subdirectories:
    - **cognate** Contains the parallel data from which cognates are extracted. The path to this directory is set to *COGNATE_DIR* in *train_SC.sh*. The src and tgt parallel data are saved to files *{COGNATE_DIR}/train.{SRC}* and *{COGNATE_DIR}/train.{TGT}*, as explained in **2.2**.
    - **fastalign** This is where the Fast Align results and the final list of cognates extracted from the parallel data in the **cognate** subdirectory are written. The path to this directory is set to *FASTALIGN_DIR* in *train_SC.sh*. This directory is discussed in **2.3**.

#### 2.2 Gather parallel data from which cognates are extracted (Pipeline/make_SC_training_data.py)
Again, note that *PARALLEL_TRAIN*, *PARALLEL_VAL*, *PARALLEL_TEST* .csv files are define the **NMT** training, validation, and test data -- NOT training data for cognate prediction. We will extract cognates from ALL of the NMT training, validation, and testing data to create cognate prediction training data.

The *Pipeline/make_SC_training_data.py* script is a bit of a misnomer. It simply reads from the *PARALLEL_TRAIN*, *PARALLEL_VAL*, *PARALLEL_TEST* .csv files and writes the parallel data to *{COGNATE_TRAIN}/cognate/train.{SRC}* and *{COGNATE_TRAIN}/cognate/train.{TGT}*. ONLY parallel data for the provided src-tgt pair through *--src* and *--tgt* commandline arguments is written. Other pairs in the .csvs, if they exist, are ignored.

#### 2.3 Run Fast Align
Now that we have written all of our parallel data to files, we can run it through Fast Align to get word pair alignments.

##### 2.3.1
Here, we create our file paths for our aligned word list files, depending on whether *NO_GROUPING* is True / False. *NO_GROUPING* should probably be True.

##### 2.3.2 (word_alignments/prepare_for_fastalign.py) 
We need to format the inputs for fast_align. This is done by the *word_alignments/prepare_for_fastalign.py* script. 

The input files to this script are the output files from *Pipeline/make_SC_training_data.py*, *i.e.,* *{COGNATE_TRAIN}/cognate/train.{SRC}* and *{COGNATE_TRAIN}/cognate/train.{TGT}*. 

This script will write the result to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.txt*, which writes each sentence pair to a line in the format ```{source sentence} ||| {target sentence}```. 

If *REVERSE_SRC_TGT_COGNATES* is set to *false*, then the source and target sentences will be flipped: ```{target sentence} ||| {source sentence}```. This setting, however, should **not** be used. Keep *REVERSE_SRC_TGT_COGNATES* set to *true*.

##### 2.3.3 Fast Align
Here we run Fast Align on the parallel sentences to get aligned word pairs. We want the symmetricized alignment, so we have to run a forward and reverse alignment first, that is, we run three Fast Align commands: (1) forward alignment, (2) reverse alignment, (3) retrieving a symmetricized alignment from the forward and reverse alignments.

Forward alignment


### 3) TRAIN SC MODEL WITH COPPER MT