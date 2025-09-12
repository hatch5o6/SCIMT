# CharLOTTE
This is the code base for **CharLOTTE**, a system that leverages character correspondences between related languages in low-resource NMT. 

**CharLOTTE** stands for **Char**acter-**L**evel **O**rthographic **T**ransfer for **T**oken **E**mbeddings.

The CharLOTTE system assumes that the phenomenon of systematic sound correspondence in linguistics is reflected in character correspondences in orthography. For example, *j-lh* and *h-f* correspondences between Spanish and Portugues, seen in word pairs: 
- *ojo, olho*
- *ajo, alho*
- *hierro, ferro*
- *horno, forno* 
- *hijo , filho*

CharLOTTE detects these character correspondences and trains tokenizers and NMT systems that exploit them so as to increase vocabulary overlap between related high and low-resourced languages. CharLOTTE utilizes a language-agnostic approach, requiring only the NMT parallel training, validation, and testing data; though additional sets of known langauge-specific sets of cognates can also be provided.


# Installation
## Clone CopperMT and add new/updated scripts
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
This documentation is designed to walk you through the *Pipeline/train_SC.sh* script. You should read this documentation and the *train_SC.sh* script together. This documentation will refer to sections of the *train_SC.sh* code with numbers like 2.2 and 2.3.1.

**Pipeline/train_SC.sh** trains the character correspondence (SC) models.
We call it SC, which stands for "sound correspondence", but more accurately, what we're detecting are character correspondences.

**Pipeline/train_SC.sh** is run from /Cognate/code, and takes a single positional argument, one of the *.cfg* config files described above, e.g.:
```
bash Pipeline/train_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC/fr-mfe.cfg
```

**Parallel Data .csv files** - *.csv* files defining the NMT parallel training, validation, and test data are referenced in the *.csg* config files and this script. These files **MUST** contain the header ```src_lang, tgt_lang, src_path, tgt_path``` where:
    - **src_lang** is the source language code
    - **tgt_lang** is the target language code
    - **src_path** is the path to the source parallel data text file
    - **tgt_path** is the path to the target parallel data text file

*src_path* and *tgt_path* must be parallel to each other, with *src_path* containing one sentence per line and *tgt_path* containing the corresponding translations on each line.


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

**Pipeline/make_SC_training_data.py**
- *--train_csv:* Parallel Data *.csv* file defining the NMT training data.
- *--val_csv:* Parallel Data *.csv* file defining the NMT validation data.
- *--test_csv:* Parallel Data *.csv* file defining the NMT test data.
- *--src:* the source language code
- *--tgt:* the target language code
- *--src_out:* the file path of the source sentences of the parallel data from which cognates will be extracted. Should be *{COGNATE_TRAIN}/cognate/train.{SRC}*.
- *--tgt_out:* the file path of the target sentences of the parallel data from which cognates will be extracted. Should be *{COGNATE_TRAIN}/cognate/train.{TGT}*.

#### 2.3 Run Fast Align
Now that we have written all of our parallel data to files, we can run it through Fast Align to get word pair alignments.

###### 2.3.1
Here, we create our file paths for our aligned word list files, depending on whether *NO_GROUPING* is True / False. *NO_GROUPING* should probably be True. These files are discussed in **2.4.1** and **2.4.2**.

###### 2.3.2 (word_alignments/prepare_for_fastalign.py) 
We need to format the inputs for fast_align. This is done by the *word_alignments/prepare_for_fastalign.py* script. 

The input files to this script are the output files from *Pipeline/make_SC_training_data.py*, *i.e.,* *{COGNATE_TRAIN}/cognate/train.{SRC}* and *{COGNATE_TRAIN}/cognate/train.{TGT}*. 

This script will write the result to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.txt*, which writes each sentence pair to a line in the format ```{source sentence} ||| {target sentence}```. 

If *REVERSE_SRC_TGT_COGNATES* is set to *false*, then the source and target sentences will be flipped: ```{target sentence} ||| {source sentence}```. This setting, however, should **not** be used. Keep *REVERSE_SRC_TGT_COGNATES* set to *true*.

**word_alignments/prepare_for_fastalign.py**
* *--src:* The file to the source parallel data from which cognates will be extracted. Should be *{COGNATE_TRAIN}/cognate/train.{SRC}*.
* *--tgt:* The file to the target parallel data from which cognates will be extracted. Should be *{COGNATE_TRAIN}/cognate/train.{TGT}*.
* *--out:* The path to the formatted sentence pairs. Should be *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.txt*.

###### 2.3.3 Fast Align
Here we run Fast Align on the parallel sentences to get aligned word pairs. We want the symmetricized alignment, so we have to run a forward and reverse alignment first, that is, we run three Fast Align commands: (1) forward alignment, (2) reverse alignment, (3) retrieving a symmetricized alignment from the forward and reverse alignments (using *grow-diag-final-and* algorithm).

Forward alignment is saved to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.forward.align*
Reverse alignment is saved to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.reverse.align*
Symmetricized alignment is saved to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.sym.align*

#### 2.4 Get Cognates
###### 2.4.1 Get word alignments (make_word_alignments(_no_grouping).py)
We then need to extract the word pairs from the Fast Align results, which is done with either the *word_alignments/make_word_alignments_no_grouping.py* or *word_aligments/make_word_alignments.py* scripts, depending on if *NO_GROUPING* is set to *true* or *false*. It should probably be set to *true*.

In essence, these two scripts read the word-level alignments from the symmetricized Fast Align results (*{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.sym.align*) and retrieve the corresponding word pairs.

The *make_word_alignments_no_grouping.py* version (the one that should probably be used) of the script simply grabs the word pair for each *i-j* pair in the alignment results where *i* is the index of a word in a source line and *j* is the index of a word in the target line.

The *make_word_alignments.py* script adds grouping logic when there are many-to-one, one-to-many, and many-to-many alignments, essentially creating phrase pairs rather than word pairs, where applicable. We should probably not use this script, for simplicity. Evaluating whether it improves performance is more complexity than I want to add right now.

These scripts write a list of source-target word pairs in the format ```{source_word} ||| {target word}```.  to *make_word_alignments_no_grouping.py* writes the results to *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}.NG.txt* (note the NG), whereas *make_word_alignments.py* writes to *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}.txt* (note absence of NG). These paths are set in the code of section **2.3.1**.

**word_alignments/make_word_alignments(_no_grouping).py**
* *--alignments, -a:* The path to the Fast Align symmetricized results. Should be *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.sym.align*.
* *--sent_pairs, -s:* The path to the sentence pairs. Should be the same as the outputs of *word_alignments/prepare_for_fastalign.py* and inputs to Fast Align in **2.3.3**, *i.e.,* should be *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.txt*
* *--out, -o:* The output path to the aligned word pairs. Should be *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).txt*.
* *--VERBOSE (optional):* Pass this flag to for verbose print outs.
* *--START (int, optional):* (make_word_alignments.py ONLY) If passed, this slices the list of sentence pairs from which to retrieve aligned words pairs to those starting with the provided START index (includes the START index). (Start index of sentences).
* *--STOP (int, optional):* If passed, this slices the list of sentence pairs from which to retrieve aligned word pairs to those up to the provided STOP index (excludes the STOP index). (Stop index of sentences).

##### 2.4.2 Get cognates from word list (word_alignments/make_cognate_list.py)

We now will narrow down the list of aligned word pairs to a list of cognate predictions by filtering the list to those pairs within a normalized edit distance threshold (*COGNATE_THRESH*). 

This is done with *word_alignments/make_cognate_list.py*. This calculates the normalized levenshtein distance of each word pair and for pairs whose distance are less than or equal to the threshold (default = 0.5), the pair of words are considered cognates.

The list of cognate pairs are written in the format ```{word 1} ||| {word 2}``` to *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.txt*. Additionally, parallel files of the source and target language words are written to *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.parallel-{SRC}.txt* and *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.parallel-{TGT}.txt*. These paths are set in the code of section **2.3.1**.

**word_alignments/make_cognate_list.py**
* *--word_list, -l:* The list of word pairs. This should be the output of *word_alignments/make_word_alignments(_no_grouping).py*, that is, it should be *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).txt*.
* *--theta, -t (float):* This is the normalized levenshtein distance threshold. Word pairs with a normalized distance less than or equal to this value will be considered cognates.
* *--src:* The source language code.
* *--tgt:* The target language code.
* *--out, -o (optional):* Path where the final cognate pairs wil be written. If not passed, will be written to *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.txt*. Parallel source and target cognate files will be written to files of the same path, except ending in *.parallel-{SRC}.{file extension}* and *.parallel-{TGT}.{file extension}*.


### 3) TRAIN SC MODEL WITH COPPER MT

#### 3.1 Make cognate prediction training, validation, and test sets

###### 3.1.1 If needed, make dataset splits
If datasets for cognate prediction validation and testing are not provided in the *.cfg* config file with *VAL_COGNATES_SRC*, *VAL_COGNATES_TGT*, *TEST_COGNATES_SRC*, *TEST_COGNATES_TGT*, then the cognate word pairs extracted from the parallel data will be divided into training, validation, and testing sets. The *train_SC.sh* script checks if this needs to be done by checking if *TEST_COGNATES_SRC* equals "null".

If *TEST_COGNATES_SRC* equals "null", then the script *Pipeline/split.py* is run to make the train, validation, and test splits on the detected cognates. This script writes the split data to files in the pattern *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.parallel-({SRC}|{TGT}).(train|test|val)-s={SEED}.txt*. In total, there are six files: a source file and a target file for each of the train, validation, and test sets.


These six files are saved to the following variables in *train_SC.sh*:
- TRAIN_COGNATES_SRC
- TRAIN_COGNATES_TGT
- VAL_COGNATES_SRC - overwriting the value set in the *.cfg* config file, which should have been "null"
- VAL_COGNATES_TGT - overwriting the value set in the *.cfg* config file, which should have been "null"
- TEST_COGNATES_SRC - overwriting the value set in the *.cfg* config file, which should have been "null"
- TEST_COGNATES_SRC - overwriting the value set in the *.cfg* config file, which should have been "null"

**Pipeline/split.py**
* *--data1:* Path to the words in the source language. Should be *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.parallel-{SRC}.txt*. (WORD_LIST_SRC path set in **2.3.1**)
* *--data2:* Path to the corresponding cognate words in the target language. Should be *{COGNATE_TRAIN}/fastalign/word_list.{SRC}-{TGT}(.NG).cognates.{COGNATE_THRESH}.parallel-{TGT}.txt*. (WORD_LIST_TGT path set in **2.3.1**)
* *--train (float):* The ratio of cognate pairs to put in the training data. *--train + --val + --test* must equal 1.
* *--val (float):* The ratio of cognate pairs to put in the validation data. *--train + --val + --test* must equal 1.
* *--test (float):* The ratio of congate pairs to put in the test data. *--train + --val + --test* must equal 1.
* *--seed (int):* The seed for random shuffling.
* *--out_dir:* Directory where output files are saved. Should be *{COGNATE_TRAIN}/fastalign*. File names of output will be same as --data1 and data2, but in the provided directory, and with an ammended extension *.(train|val|test)-s={SEED}.{original file extension}*.
* *--UNIQUE_TEST:* If this flag is passed, then will reduce the test set so that a given source word only occurs once.

If dataset splits don't need to be added, meaning *TEST_COGNATES_SRC* is not "null", then all of *VAL_COGNATES_SRC*, *VAL_COGNATES_TGT*, *TEST_COGNATES_SRC*, *TEST_COGNATES_TGT* should be set (**not** "null") in the *.cfg* config file to files containing known cognates, such as from Cognet and/or EtymDB. In this case, these files will be used for validation and testing and *TRAIN_COGNATES_SRC* and *TRAIN_COGNATES_TGT* will be set to files containing all of the cognate pairs detected from the parallel NMT data.

###### 3.1.2 Include ADDITIONAL_TRAIN_COGNATES_SRC and ADDITIONAL_TRAIN_COGNATES_TGT in train set file paths

Files containing known cognate pairs, such as from CogNet and EtymDB, can also be set to *ADDITIONAL_TRAIN_COGNATES_SRC* and *ADDITIONAL_TRAIN_COGNATES_TGT*. If so, these will be appended to *TRAIN_COGNATES_SRC* and *TRAIN_COGNATES_TGT* as a comma-delimited list.

###### 3.1.3 Print out files for train, validation, and test sets
The comma-delimited lists of files in *TRAIN_COGNATES_SRC*, *TRAIN_COGNATES_TGT*, *VAL_COGNATES_SRC*, *VAL_COGNATES_TGT*, *TEST_COGNATES_SRC*, *TEST_COGNATES_TGT* are printed.

#### 3.2 Train CopperMT cognate prediction model

###### 3.2.1 make directory structure for CopperMT scenario inputs and outputs
The directory structure for the CopperMT scenario is created. This structure will contain the model, training data, outputs, etc. If the parent directory of this structure already exists, it will be deleted then recreated.

The parent directory should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}*. 

###### 3.2.2 Copy the RNN hyperparams set file, corresponding to RNN_HYPERPARAMS_ID, to its place in the COPPERMT directory structure
The RNN hyperparameters file corresponding to *RNN_HYPERPARAMS_ID* is copied to its place in the CopperMT scenario directory structure.

**Pipeline/copy_rnn_hyperparams.py**
* *--rnn_hyperparam_id, -i:* The ID of the desired RNN hyperparam set.
* *--rnn_hyperparams_dir, -d:* Folder containing the RNN hyperparam set files. Should be *RNN_HYPERPARAMS*.
* *--copy_to_path, -c:* The path the RNN hyperparams set file will be copied to. Should be copied to the appropriate place inside the CopperMT scenario directory structure: *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/parameters/bilingual_default/default_parameters_rnn_{SRC}-{TGT}.txt*.

###### 3.2.3 Format the cognate train, val, test data for CopperMT
The cognate pair data needs to be formatted for the CopperMT module. This is done with *CopperMT/format_data.py*, which is run three times: once each for the training, validation, and test data sets. This script takes the parallel cognate files, and writes the cognate pairs in the CopperMT format to files in the CopperMT scenario directory structure. Specifically, they will be written to the folder *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}*. Parallel cognate files for training are called *train_{SRC}_{TGT}.{SRC}* and *train_{SRC}_{TGT}.{TGT}*, for validation, *fine_tune_{SRC}_{TGT}.{SRC}* and *fine_tune_{SRC}_{TGT}.{TGT}*, and for testing, *test_{SRC}_{TGT}.{SRC}* and *test_{SRC}_{TGT}.{TGT}*. (NOTE, the *fine_tune* prefix was established by CopperMT module, but is actually used to refer to validation data). the *CopperMT/format_data.py* script will also shuffle each dataset and make sure it (internally) has only unique source-target cognate pairs.

**CopperMT/format_data.py**
* *--src_data (str):* comma-delimited list of parallel source cognate files. Should be variabel *TRAIN/VAL/TEST_COGNATES_SRC*.
* *--tgt_data (str):* comma-delimited list of parallel target cognate files, corresponding to those passed to *--src_data*. Should be variabel *TRAIN/VAL/TEST_COGNATES_TGT*.
* *--src (str):* Source language code.
* *--tgt (str):* Target language code.
* *--out_dir (str):* The directory the formatted output files will be written to. Note that the files will be written to a subdirectory of this directory corresponding to the seed (see *--seed* below). This should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}*, and hence, the files will be written to *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}*.
* *--prefix (str):* Must be "train", "fine_tune", or "test", depending on if it's the training, validation, or test set (use "fine_tune" for validation set).
* *--seed (int):* The seed to use for random shuffling of the data. Will also be the name of the subdirectory the output files will be in.

###### 3.2.4 Assert there is no overlap of src and tgt segments (words) between the cognate prediction train / dev / test data
Here we just make sure there no source or target words overlapping between the cognate prediction train, dev, and test datasets. More than ensure there are no overlapping pairs, this ensures there are no overlapping source words or overlapping target words.

The *CopperMT/assert_no_overlap_in_formatted_data* script is run twice to do this. The first time (without the --TEST_ONLY flag), it will remove any existing overlap between the train, dev, and test sets. It does this by first checking if any source words in the train exist in the source side of either the dev or test, and removes the corresponding pairs. It then does the same for target words, checking if any exist in the target side of the dev or test set, and removing corresponding pairs. This process is repeated for the dev set, though now only checking if the words exist in the test set.

On the second run, it will simply just test, mostly for good measure, that there are no overlapping source or target words accross train, dev, and test sets.

**CopperMT/assert_no_overlap_in_formatted_data**
* *--format_out_dir:* The directory the formatted data is written to. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}*.
* *--src* Source language code.
* *--tgt* Target language code.
* *--TEST_ONLY* If this flag is passed, it will ONLY check that there is no overlap between the train, fine_tune (validation), and test sets. If it is not passed, then the script will remove any existing overlap.


###### 3.2.5 Log the cognate predition data
First, the a log .json file is chosen, depending on if NO_GROUPING is true or false (should be true).

Then the sizes of the train, val, and test (for the corresponding language) sets are logged to the log file. This log file maintains a history. See the "latest" key for the latest logged sizes, and "history" for the history of the size change. A corresponding .csv file (same path as the .json log file, but with a .csv extension) is also written, which just shows the latest sizes.

**Pipeline/cognate_dataset_log.py**
* *--formatted_data_dir, -f:* The formatted data directory. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}*. 
* *--lang_pair, -l:* The source-target language pair, formatted "{source lang}-{target lang}". Should be *{SRC}-{TGT}*.
* *--LOG_F, -L:* the path of the .json log to be updated. Will be either *cognate_dataset_log_NG=True.json* or *cognate_dataset_log_NG=False.json*, depending on the value of *NO_GROUPING* (which should be true). A corresponding .csv file will also be written.

###### 3.2.6 Write the CopperMT parameters file
The parameters file required by the CopperMT module needs to be written, which is performed by *Pipeline/write_scripts.py*.

**Pipeline/write_scripts.py**
* *--src:* Source language code. 
* *--tgt:* Target language code
* *--coppermt_data_dir:* Parent folder containing the training data, models, and outputs of each cognate prediction scenario. Should be *COPPERMT_DATA_DIR*.
* *--sc_model_type:* The SC model type, either "RNN" or "SMT". Should be *SC_MODEL_TYPE*.
* *--rnn_hyperparams_id:* The id corresonding to the desired RNN hyperparams set. Should be *RNN_HYPERPARAMS_ID*.
* *--seed:* Should be *SEED*.
* *--parameters, -p:* The path the CopperMT parameters file will be written to. Should be *{PARAMETERS_DIR}/parameters.{SRC}-{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}.cfg*

###### 3.2.7 Train the SC model with CopperMT
The SC model is now trained. This is done by calling scripts in the CopperMT module.

**Training an RNN model:**  
If training an RNN model, *{COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan.sh* script is called, passing in the parameters file created in **3.2.6** (should be *{PARAMETERS_DIR}/parameters.{SRC}-{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}.cfg*) and the *SEED*. 

After training, the best RNN checkpoint is selected, using *Pipeline/select_checkpoint.py*. This selects the best performing checkpoint, based on BLEU score calculated by CopperMT, from those in a directory that contains checkpoints and outputs. This directory is set to variable *WORKSPACE_SEED_DIR*, which should be *COPPERMT_DATA_DIR/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}*. *Pipeline/select_checkpoint.py* will save the best checkpoint to *{WORKSPACE_SEED_DIR}/checkpoints/selected.pt*. All other checkpoints will be deleted to conserve storage space.

**Training an SMT model:**  
If training an SMT model, *{COPPERMT_DIR}/pipeline/main_smt_full_brendan.sh* is run, passing the same parameters file from **3.2.6** and *SEED*.



### 4) EVALUATE SC MODEL

#### 4.1 Delete inference directories if pre-existing
A couple directories, if pre-existing, are deleted.
#TODO describe what they are after you figure this out. (Are they still used or did we change this?? The files aren't called elsewhere in train_SC.sh or pred_SC.sh). Run the script, and see afterwards if they exist.

#### 4.2 Run inference on the test set
To calculate scores, inference is first run on the test set.

**Inference with an RNN model**  
To run inference with an RNN model, *{COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan_PREDICT.sh* is called, passing the Copper MT parameters file from **3.2.6** (*PARAMETERS_F*), the path to the selected RNN checkpoint from **3.2.7** (*SELECTED_RNN_CHECKPOING*), *SEED*, an indicator "test", *NBEST*, and *BEAM*. This script will save its results to a file whose path is saved to the variable *HYP_OUT_TXT*. This path should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/results/test_selected_checkpoint_{SRC}_{TGT}.{TGT}/generate-test.txt*.

The model hypotheses need to be extracted from the *HYP_OUT_TXT* file, which is done with the **NMT/hr_CopperMT.py** script. This script has three modes: "prepare", "retrieve", and "get_test_results". Modes "prepare" and "retrieve" will be discussed later in connection to *pred_SC.sh*. To extract the hypotheses from the model test results file, we use mode "get_test_results". Only the parameters relevant to this mode are shown here. This mode will write the hypotheses to a file parallel to the source file, where on each line is simply the cognate hypothesis for each source word.

**NMT/hr_CopperMT.py (get_test_results)**
* *--function* The script mode. In this case, it should be "get_test_results".
* *--test_src:* The test source sentences. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}/test_{SRC}_{TGT}.{SRC}* (saved to variable *SRC_TEXT*).
* *--data:* The model results, written by *main_nmt_bilingual_full_brendan_PREDICT.sh*. The path is saved to *HYP_OUT_TXT*.
* *--out:* The path to save the hypotheses extracted from the model results file. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/results/test_selected_checkpoint_{SRC}_{TGT}.{TGT}/generate-test.hyp.txt* (saved to *TEST_OUT_F*).

The path to write the scores for an RNN model (set to variable *SCORES_OUT_F*) is then set to *{COPPERMT_DATA_DIR}/{SRC}_${TGT}_${SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/results/test_selected_checkpoint_{SRC}_{TGT}.{TGT}/generate-test.hyp.scores.txt*. This path will be used in **4.3**.

**Inference with an SMT model**  
To run inference with an SMT model, *{COPPERMT_DIR}/pipeline/main_smt_full_brendan_PREDICT.sh* is run, passing in the Copper MT parameters file from **3.2.6** (*PARAMETERS_F*), the file path of the source sentences (*SRC_TEXT*), a template for the outputs (*HYP_OUT*), and *SEED*. The hypotheses will be written to *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}/test_{SRC}_{TGT}.{TGT}.hyp.txt* (saved to variable *TEST_OUT_F*).

The path to write the scores for an SMT model (set to variable *SCORES_OUT_F*) is then set to *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}/test_{SRC}_{TGT}.{TGT}.hyp.scores.txt*. This path will be used in **4.3**.

#### 4.3 Calculate scores
Finally, the results are evaluated using *NMT/evaluate.py* which will calculate a character-level BLEU score (actually just regular BLEU, but since characters in the output are separated by spaces, it amounts to character-level BLEU), and chrF.

**NMT/evaluate.py**
* *--ref:* The path to the reference translations. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}/test_{SRC}_{TGT}.{TGT}*
* *--hyp:* The path to the model hypotheses, saved to *TEST_OUT_F*, set in **4.2**.
* *--out:* The file path to write the scores to, which is *SCORES_OUT_F*, set in **4.2**.

