# CharLOTTE
This is the code base for **CharLOTTE**, a system that leverages character correspondences between related languages in low-resource NMT. 

**CharLOTTE** stands for **Char**acter-**L**evel **O**rthographic **T**ransfer for **T**oken **E**mbeddings.

The CharLOTTE system assumes that the phenomenon of systematic sound correspondence in linguistics is reflected in character correspondences in orthography. For example, *j-lh* and *h-f* correspondences between Spanish and Portugues, seen in word pairs: 
- *ojo, olho*
- *ajo, alho*
- *hierro, ferro*
- *horno, forno* 
- *hijo , filho*

CharLOTTE learns these character correspondences with we call **SC models** and trains tokenizers and NMT models that exploit them so as to increase vocabulary overlap between related high and low-resourced languages. CharLOTTE utilizes a language-agnostic approach, requiring only the NMT parallel training, validation, and testing data; though additional sets of known langauge-specific sets of cognates can also be provided.

## What are SC Models?

**SC** stands for **Sound Correspondence** (though more accurately, "character correspondence" since the system operates on orthography rather than phonetic transcriptions).

SC models learn systematic character-level mappings between related languages. CharLOTTE uses these models to address a fundamental challenge in low-resource NMT:

**The Challenge**: Training high-quality NMT requires large parallel datasets, but low-resource languages have limited data.

**The SC Solution**:
1. Identify a high-resource language related to your low-resource target (e.g., Spanish for Aragonese, French for Mauritian Creole)
2. Train an **SC model** to learn character correspondences between the high-resource and low-resource languages
3. **Apply the SC model** to transform high-resource parallel data, making it orthographically similar to the low-resource language
4. Train NMT using both the original low-resource data AND the SC-normalized high-resource data

**Example**: For Aragonese→English NMT with limited Aragonese-English data:
- Train SC model: Spanish → Aragonese character correspondences
- Apply to data: Transform Spanish-English corpus to look like Aragonese-English
- Result: Spanish word *hijo* → Aragonese-like *fillo* (learning correspondences like *j→ll*, *i→i*, *o→o*)
- Train NMT with augmented data, benefiting from increased vocabulary overlap

**SC Model Types**:
- **RNN**: Sequence-to-sequence neural model for cognate prediction
- **SMT**: Statistical machine translation model for cognate prediction

Both are trained using the CopperMT framework and can predict character-level transformations to generate plausible cognates.

# Prerequisites

## Python Environment
This project requires Python 3.10+ and has two separate dependency sets:

1. **Sound Correspondence (SC) Models**: Install dependencies for training SC models and tokenizers:
   ```bash
   pip install -r sound.requirements.txt
   ```
   Key dependencies: PyTorch, PyTorch Lightning, transformers, SentencePiece, sacrebleu

2. **CopperMT**: Install dependencies for the CopperMT cognate mining module:
   ```bash
   pip install -r copper.requirements.txt
   ```
   Key dependencies: fairseq, CopperMT dependencies

## External Tools
- **FastAlign**: Required for word alignment in cognate detection. Install following instructions at: https://github.com/clab/fast_align

# Installation
## Clone CopperMT and add new/updated scripts
From root directory, run these:
```
cd CopperMT
git clone https://github.com/clefourrier/CopperMT.git
cd ../CopperMTfiles
python move_files.py
```

# Obtaining Training Data

Before running the CharLOTTE pipeline, you need parallel data for both your low-resource and high-resource language pairs.

## Data Requirements

### Minimum Dataset Sizes
For meaningful experimental results:
- **Low-resource pair**: 5,000+ sentence pairs minimum (10,000+ recommended)
- **High-resource pair**: 100,000+ sentence pairs recommended (more is better)
- **Validation set**: 500-1,000 sentence pairs per language pair
- **Test set**: 1,000-2,000 sentence pairs per language pair

### Data Sources

**Public Parallel Corpora**:
- **OPUS** (https://opus.nlpl.eu/): Largest collection of freely available parallel corpora
  - Includes: OpenSubtitles, Europarl, Wikipedia, Bible translations, etc.
  - Covers 100+ languages
  - Download in plain text or TMX format
- **Tatoeba** (https://tatoeba.org/): Sentence-level translations (good for low-resource languages)
- **JW300** (Jehovah's Witness translations): Available for many low-resource languages
- **CCMatrix**: Mined parallel sentences from CommonCrawl

**For Low-Resource Languages**:
If your target low-resource language isn't available:
1. Check linguistic resources: Bible translations, religious texts, government documents
2. Consider creating synthetic data using the SC model in reverse
3. Use web-scraped bilingual websites (with appropriate permissions)

**Example: Obtaining Aragonese-English Data**:
```bash
# Download from OPUS (example)
wget https://opus.nlpl.eu/download.php?f=Tatoeba/v2023-04-12/tmx/an-en.tmx.gz
gunzip an-en.tmx.gz

# Convert TMX to plain text (use tmx2txt or similar tool)
# Results in: an-en.an (source) and an-en.en (target)
```

## Creating Train/Val/Test Splits

Once you have parallel data, split it into train/validation/test:

```python
import random

def split_parallel_data(src_file, tgt_file, output_prefix, train_ratio=0.8, val_ratio=0.1):
    """Split parallel data into train/val/test sets."""
    # Read data
    with open(src_file) as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_file) as f:
        tgt_lines = [line.strip() for line in f]

    assert len(src_lines) == len(tgt_lines), "Source and target must have same length"

    # Shuffle together
    pairs = list(zip(src_lines, tgt_lines))
    random.seed(42)  # For reproducibility
    random.shuffle(pairs)

    # Calculate split points
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split data
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    # Write splits
    for split_name, split_pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        src_out = f"{output_prefix}.{split_name}.src"
        tgt_out = f"{output_prefix}.{split_name}.tgt"

        with open(src_out, 'w') as f:
            f.write('\n'.join([s for s, t in split_pairs]) + '\n')
        with open(tgt_out, 'w') as f:
            f.write('\n'.join([t for s, t in split_pairs]) + '\n')

        print(f"{split_name}: {len(split_pairs)} pairs")
        print(f"  Written to: {src_out}, {tgt_out}")

# Example usage:
split_parallel_data('an-en.an', 'an-en.en', 'aragonese-english')
# Creates: aragonese-english.train.src, aragonese-english.train.tgt, etc.
```

## Using Your Own Language Pair

To adapt CharLOTTE to your own low-resource scenario:

1. **Identify languages**:
   - Low-resource target (e.g., Breton)
   - High-resource related language (e.g., Welsh or French)
   - Pivot language (usually English)

2. **Gather data**:
   - Low-resource ↔ pivot: Find or create parallel data
   - High-resource ↔ pivot: Download from OPUS
   - Ensure languages are actually related (same family or in contact)

3. **Verify language relationship**:
   - SC models work best for related languages (cognates exist)
   - Check: Can you find word pairs with systematic sound patterns?
   - If languages are unrelated, SC augmentation may not help

# Data Preparation

After obtaining your parallel data, prepare it in the correct format for CharLOTTE.

## Directory Structure

We recommend organizing your project with the following structure:

```
your-project/
├── data/
│   ├── raw/                          # Your raw parallel text files
│   │   ├── low-resource/
│   │   │   ├── train.src
│   │   │   ├── train.tgt
│   │   │   ├── val.src
│   │   │   ├── val.tgt
│   │   │   ├── test.src
│   │   │   └── test.tgt
│   │   └── high-resource/
│   │       ├── train.src
│   │       ├── train.tgt
│   │       └── ...
│   └── csv/                          # CSV metadata files
│       ├── train.no_overlap_v1.csv
│       ├── val.no_overlap_v1.csv
│       └── test.csv
├── models/
│   ├── sc_models/                    # SC model outputs
│   ├── tokenizers/                   # Tokenizer outputs
│   └── nmt_models/                   # NMT model outputs
└── configs/
    ├── sc/                           # SC config files
    ├── tok/                          # Tokenizer config files
    └── nmt/                          # NMT config files
```

## Parallel Data Format

### Raw Text Files

Parallel text files must be sentence-aligned with one sentence per line:

**Example: train.src (Aragonese)**
```
Iste ye un exemplo en aragonés.
O tiempo ye bueno güei.
```

**Example: train.tgt (English)**
```
This is an example in Aragonese.
The weather is good today.
```

Each line in the source file must correspond to the translation on the same line number in the target file.

### CSV Metadata Files

The CharLOTTE pipeline uses CSV files to specify parallel data locations. These CSV files have the header:

```
src_lang,tgt_lang,src_path,tgt_path
```

**Example: data/csv/train.no_overlap_v1.csv**
```csv
src_lang,tgt_lang,src_path,tgt_path
an,en,/absolute/path/to/data/raw/low-resource/train.src,/absolute/path/to/data/raw/low-resource/train.tgt
es,en,/absolute/path/to/data/raw/high-resource/train.src,/absolute/path/to/data/raw/high-resource/train.tgt
```

**Important Notes**:
- Paths must be **absolute paths**, not relative
- Training CSV must be named `train.no_overlap_v1.csv`
- Validation CSV must be named `val.no_overlap_v1.csv`
- Test CSV must be named `test.csv`
- You can include multiple language pairs in a single CSV (for multilingual training)
- The `no_overlap_v1` naming convention indicates train/val splits have no overlapping sentences

### Creating CSV Files

Here's a Python script to generate CSV files from your raw data:

```python
import csv
import os

# Configuration
BASE_DIR = "/absolute/path/to/your-project"
DATA_DIR = f"{BASE_DIR}/data"
RAW_DIR = f"{DATA_DIR}/raw"
CSV_DIR = f"{DATA_DIR}/csv"

# Language pair configuration
LOW_RESOURCE_SRC = "an"  # Aragonese
LOW_RESOURCE_TGT = "en"  # English
HIGH_RESOURCE_SRC = "es"  # Spanish
HIGH_RESOURCE_TGT = "en"  # English

# Create CSV directory
os.makedirs(CSV_DIR, exist_ok=True)

# Define splits
splits = [
    ("train.no_overlap_v1.csv", "train"),
    ("val.no_overlap_v1.csv", "val"),
    ("test.csv", "test")
]

for csv_filename, split in splits:
    csv_path = f"{CSV_DIR}/{csv_filename}"

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['src_lang', 'tgt_lang', 'src_path', 'tgt_path'])

        # Low-resource data
        writer.writerow([
            LOW_RESOURCE_SRC,
            LOW_RESOURCE_TGT,
            f"{RAW_DIR}/low-resource/{split}.src",
            f"{RAW_DIR}/low-resource/{split}.tgt"
        ])

        # High-resource data (usually only for train)
        if split == "train":
            writer.writerow([
                HIGH_RESOURCE_SRC,
                HIGH_RESOURCE_TGT,
                f"{RAW_DIR}/high-resource/{split}.src",
                f"{RAW_DIR}/high-resource/{split}.tgt"
            ])

    print(f"Created: {csv_path}")
```

Run this script after placing your parallel text files in the `data/raw/` directory structure.

# Quick Start

This quick start demonstrates the complete CharLOTTE pipeline for low-resource NMT with a concrete example.

## Prerequisites
Ensure you have:
- Installed all dependencies (see [Prerequisites](#prerequisites))
- Installed FastAlign and verified it's in your PATH: `which fast_align`
- Prepared your parallel data (see [Data Preparation](#data-preparation))

## Quick Test: Verifying Your Setup (15 minutes)

Before running the full pipeline, test your setup with a minimal dataset to catch any configuration issues early.

### Create Toy Dataset

```bash
# Create tiny parallel data (100 sentences each)
mkdir -p ~/charlotte-test/data/{raw,csv}

# Generate toy Aragonese-English data
cat > ~/charlotte-test/data/raw/train.an <<EOF
Iste ye un exemplo.
O tiempo ye bueno.
Yo parllo aragonés.
EOF

cat > ~/charlotte-test/data/raw/train.en <<EOF
This is an example.
The weather is good.
I speak Aragonese.
EOF

# Create CSV (same file for train/val/test for quick test)
cat > ~/charlotte-test/data/csv/train.no_overlap_v1.csv <<EOF
src_lang,tgt_lang,src_path,tgt_path
an,en,~/charlotte-test/data/raw/train.an,~/charlotte-test/data/raw/train.en
EOF

cp ~/charlotte-test/data/csv/train.no_overlap_v1.csv ~/charlotte-test/data/csv/val.no_overlap_v1.csv
cp ~/charlotte-test/data/csv/train.no_overlap_v1.csv ~/charlotte-test/data/csv/test.csv
```

### Test SC Training

```bash
# Create minimal SC config
cat > ~/charlotte-test/test-sc.cfg <<EOF
MODULE_HOME_DIR=$SCIMT_DIR
SRC=an
TGT=an
SEED=1000
PARALLEL_TRAIN=~/charlotte-test/data/csv/train.no_overlap_v1.csv
PARALLEL_VAL=~/charlotte-test/data/csv/val.no_overlap_v1.csv
PARALLEL_TEST=~/charlotte-test/data/csv/test.csv
APPLY_TO=~/charlotte-test/data/csv/train.no_overlap_v1.csv
NO_GROUPING=true
SC_MODEL_TYPE=SMT
COGNATE_THRESH=0.5
COGNATE_TRAIN=~/charlotte-test/cognates
COPPERMT_DATA_DIR=~/charlotte-test/sc_models
COPPERMT_DIR=$SCIMT_DIR/CopperMT/CopperMT
PARAMETERS_DIR=~/charlotte-test/params
REVERSE_SRC_TGT_COGNATES=false
SC_MODEL_ID=test
ADDITIONAL_TRAIN_COGNATES_SRC=null
ADDITIONAL_TRAIN_COGNATES_TGT=null
VAL_COGNATES_SRC=null
VAL_COGNATES_TGT=null
TEST_COGNATES_SRC=null
TEST_COGNATES_TGT=null
COGNATE_TRAIN_RATIO=0.8
COGNATE_VAL_RATIO=0.1
COGNATE_TEST_RATIO=0.1
EOF

# Run SC training (should complete in ~2 minutes with SMT)
cd $SCIMT_DIR
bash Pipeline/train_SC.sh ~/charlotte-test/test-sc.cfg
```

**Success Indicators**:
- No errors during FastAlign
- Cognate pairs extracted and saved
- SMT model trained
- Character-level BLEU/chrF scores printed

**If this works**, your environment is correctly set up for the full pipeline.

**If this fails**, check:
- FastAlign is in PATH: `which fast_align`
- All paths in config are absolute (no `~` in paths, use full `/home/username/...`)
- Dependencies installed: `pip list | grep -E "(torch|transformers|sacrebleu)"`

## End-to-End Example: Aragonese→English with Spanish Augmentation

This example shows the complete workflow for training an Aragonese→English NMT system augmented with Spanish data.

**Scenario**:
- Low-resource pair: Aragonese (an) → English (en)
- High-resource related pair: Spanish (es) → English (en)
- Goal: Use SC model to transform Spanish→Aragonese, then train NMT with augmented data

### Before Starting: Path Configuration

Throughout this example, you'll need to replace placeholder paths with your actual paths:

**Placeholders to Replace**:
1. `/path/to/SCIMT` → Your SCIMT clone location
   - Example: `/home/username/projects/SCIMT` or `/Users/username/git/SCIMT`
   - Find it: Run `pwd` in your SCIMT directory after cloning

2. `$BASE_DIR` (in shell configs) → Will expand automatically if you set the environment variable
   - Example: `export BASE_DIR=/home/username/charlotte-project`

3. `$BASE_DIR` (in YAML configs) → MUST be replaced with absolute path
   - YAML doesn't expand variables
   - Wrong: `save: $BASE_DIR/models/nmt_models/an-en/PRETRAIN`
   - Right: `save: /home/username/charlotte-project/models/nmt_models/an-en/PRETRAIN`

**Quick Setup Script**:
```bash
# Set these once at the start
export SCIMT_DIR=/path/to/your/SCIMT/clone  # CHANGE THIS
export BASE_DIR=~/charlotte-project

echo "SCIMT_DIR=$SCIMT_DIR" >> ~/.bashrc
echo "BASE_DIR=$BASE_DIR" >> ~/.bashrc
```

**Verifying Your Setup**:
```bash
# Check SCIMT installation
ls $SCIMT_DIR/Pipeline/train_SC.sh  # Should exist
ls $SCIMT_DIR/CopperMT/CopperMT     # Should exist after installation

# Check data
ls $BASE_DIR/data/raw/low-resource/train.src   # Should exist
ls $BASE_DIR/data/csv/train.no_overlap_v1.csv  # Should exist

# Check FastAlign
which fast_align  # Should print path to fast_align binary
```

### Step 0: Setup Directories

```bash
# Create project structure
mkdir -p ~/charlotte-project/{data/{raw/{low-resource,high-resource},csv},models/{sc_models,tokenizers,nmt_models},configs/{sc,tok,nmt}}

# Set base directory
export BASE_DIR=~/charlotte-project
```

### Step 1: Create SC Config File

Create `$BASE_DIR/configs/sc/es2an.cfg`:

```bash
# SC Model Configuration for Spanish → Aragonese

# Module paths
MODULE_HOME_DIR=$SCIMT_DIR  # Or use your absolute path

# NMT language pairs (for reference, not used by train_SC.sh)
NMT_SRC=an
NMT_TGT=en
AUG_SRC=es
AUG_TGT=en

# SC model language pair
SRC=es
TGT=an
SEED=1000

# Data paths
PARALLEL_TRAIN=$BASE_DIR/data/csv/train.no_overlap_v1.csv
PARALLEL_VAL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
PARALLEL_TEST=$BASE_DIR/data/csv/test.csv
APPLY_TO=$PARALLEL_TRAIN,$PARALLEL_VAL,$PARALLEL_TEST

# SC model settings
NO_GROUPING=true
SC_MODEL_TYPE=RNN
COGNATE_THRESH=0.5

# Output directories
COGNATE_TRAIN=$BASE_DIR/models/sc_models/cognates
COPPERMT_DATA_DIR=$BASE_DIR/models/sc_models
COPPERMT_DIR=/path/to/SCIMT/CopperMT/CopperMT
PARAMETERS_DIR=$BASE_DIR/configs/sc/parameters

# RNN hyperparameters (if using RNN)
RNN_HYPERPARAMS=$MODULE_HOME_DIR/Pipeline/parameters/rnn_hyperparams
RNN_HYPERPARAMS_ID=0  # Uses default hyperparameters (see note below)
BEAM=5
NBEST=1

# Cognate extraction settings
REVERSE_SRC_TGT_COGNATES=false
SC_MODEL_ID=es2an-RNN-0

# Additional cognate data (set to null if not using)
ADDITIONAL_TRAIN_COGNATES_SRC=null
ADDITIONAL_TRAIN_COGNATES_TGT=null
VAL_COGNATES_SRC=null
VAL_COGNATES_TGT=null
TEST_COGNATES_SRC=null
TEST_COGNATES_TGT=null

# Cognate split ratios (used if VAL/TEST_COGNATES are null)
COGNATE_TRAIN_RATIO=0.8
COGNATE_VAL_RATIO=0.1
COGNATE_TEST_RATIO=0.1
```

### Step 2: Train SC Model

**Note on RNN Hyperparameters**: The config above uses `RNN_HYPERPARAMS_ID=0`, which references default hyperparameters stored in `Pipeline/parameters/rnn_hyperparams/`. These files should exist in your SCIMT clone. To verify:
```bash
ls $SCIMT_DIR/Pipeline/parameters/rnn_hyperparams/
# Should show: manifest.json and numbered parameter files (0, 1, 2, etc.)
```

If this directory doesn't exist, you can use SMT instead by changing `SC_MODEL_TYPE=SMT` in the config (SMT doesn't need hyperparameter files).

**Training the Model**:
```bash
cd $SCIMT_DIR
bash Pipeline/train_SC.sh $BASE_DIR/configs/sc/es2an.cfg
```

This will:
- Extract cognates from Spanish-Aragonese parallel data using FastAlign
- Train an RNN model to predict Aragonese cognates from Spanish words
- Evaluate the model on character-level BLEU and chrF

**Expected Time**:
- Cognate extraction: 5-30 minutes (depends on data size)
- RNN training: 30 minutes - 2 hours (depends on cognate pairs and GPU)
- SMT training: 5-15 minutes

**Output**: SC model at `$BASE_DIR/models/sc_models/es_an_RNN-0_S-1000/`

### Step 3: Apply SC Model to Spanish Data

```bash
bash Pipeline/pred_SC.sh $BASE_DIR/configs/sc/es2an.cfg
```

This creates new data files with SC-normalized Spanish (now looking like Aragonese). Files will have `SC_es2an-RNN-0_es2an` in their filenames.

**Output**: Normalized data files alongside originals in `data/raw/high-resource/`

### Step 4: Create Tokenizer Config

Create `$BASE_DIR/configs/tok/es-an_en.cfg`:

```bash
# Tokenizer Configuration

# Training size
SPM_TRAIN_SIZE=1000000

# Languages
SRC_LANGS=es,an
SRC_TOK_NAME=es-an
TGT_LANGS=en
TGT_TOK_NAME=en

# Data distribution (percentages must sum to 100)
DIST=es:40,an:10,en:50

# Parallel data CSVs
TRAIN_PARALLEL=$BASE_DIR/data/csv/train.no_overlap_v1.csv
VAL_PARALLEL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
TEST_PARALLEL=$BASE_DIR/data/csv/test.csv

# Output directory
TOK_TRAIN_DATA_DIR=$BASE_DIR/models/tokenizers

# SC model ID (use if training on SC-normalized data)
SC_MODEL_ID=es2an-RNN-0-RNN-0

# Tokenizer settings
VOCAB_SIZE=32000
SPLIT_ON_WS=false
INCLUDE_LANG_TOKS=true
INCLUDE_PAD_TOK=true
SPECIAL_TOKS=null
IS_ATT=false
```

### Step 5: Train Tokenizer

```bash
bash Pipeline/train_srctgt_tokenizer.sh $BASE_DIR/configs/tok/es-an_en.cfg
```

**Output**: SentencePiece model at `$BASE_DIR/models/tokenizers/es-an_en/`

### Step 6: Create NMT Config

Create `$BASE_DIR/configs/nmt/an-en.PRETRAIN.yaml`:

```yaml
# outputs
src: an
tgt: en
save: $BASE_DIR/models/nmt_models/an-en/PRETRAIN
test_checkpoint: null
remove_special_toks: true
verbose: false
little_verbose: true

# finetune?
from_pretrained: null

# data
train_data: $BASE_DIR/data/csv/train.no_overlap_v1.csv
val_data: $BASE_DIR/data/csv/val.no_overlap_v1.csv
test_data: $BASE_DIR/data/csv/test.csv
append_src_token: false
append_tgt_token: false
upsample: false
sc_model_id: es2an-RNN-0-RNN-0

# tokenizers
spm: $BASE_DIR/models/tokenizers/es-an_en/es-an_en
do_char: false

# training
n_gpus: 1
seed: 1000
max_steps: 50000
train_batch_size: 32
val_batch_size: 32
test_batch_size: 32
early_stop: 10
save_top_k: 5
val_interval: 0.5
learning_rate: 2e-04
weight_decay: 0.01
device: cuda

# model architecture
encoder_layers: 6
encoder_attention_heads: 8
encoder_ffn_dim: 2048
encoder_layerdrop: 0.0
decoder_layers: 6
decoder_attention_heads: 8
decoder_ffn_dim: 2048
decoder_layerdrop: 0.0
max_position_embeddings: 512
max_length: 512
d_model: 512
dropout: 0.1
activation_function: gelu
```

**Note**: Replace `$BASE_DIR` with your actual absolute path in the YAML file.

### Step 7: Train NMT Model

```bash
cd $SCIMT_DIR/NMT
python train.py -c $BASE_DIR/configs/nmt/an-en.PRETRAIN.yaml -m TRAIN
```

Training will:
- Use both Aragonese-English and SC-normalized Spanish-English data
- Save checkpoints every validation interval
- Apply early stopping based on validation loss

**Expected Time & Resources**:
- Training time: 4-48 hours (depends on data size, GPU, max_steps)
  - 50k steps with 32 batch size on 1 GPU: ~6-12 hours
  - 250k steps with 128 batch size on 4 GPUs: ~24-48 hours
- GPU memory: 8GB+ recommended (reduce batch_size if OOM errors occur)
- Disk space: 2-5GB per model (checkpoints + logs)
- CPU cores: 4+ recommended for data loading

**Monitoring Training**:
```bash
# Watch training progress in real-time
tail -f $BASE_DIR/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/logs/version_0/metrics.csv

# Or use TensorBoard
tensorboard --logdir $BASE_DIR/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/logs
```

**Output**: Model checkpoints at `$BASE_DIR/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/checkpoints/`

### Step 8: Evaluate NMT Model

```bash
python train.py -c $BASE_DIR/configs/nmt/an-en.PRETRAIN.yaml -m TEST
```

**Output**:
- Test predictions: `$BASE_DIR/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/predictions/`
- BLEU and chrF scores: `metrics.json` in predictions directory

### Step 9: Visualize Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training metrics
metrics = pd.read_csv(f'{BASE_DIR}/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/logs/version_0/metrics.csv')

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(metrics['step'], metrics['train_loss_step'], label='Train Loss', alpha=0.6)
plt.plot(metrics['step'], metrics['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.show()

# Load test metrics
import json
with open(f'{BASE_DIR}/models/nmt_models/an-en/PRETRAIN_TRIAL_s=1000/predictions/all_scores.json') as f:
    scores = json.load(f)
    print(f"Best BLEU: {scores['BEST_BLEU_CHECKPOINT']['BLEU']:.2f}")
    print(f"Best chrF: {scores['BEST_BLEU_CHECKPOINT']['chrF']:.2f}")

# Optional: Compute COMET score (see "Optional: COMET Evaluation" section)
# Requires installing unbabel-comet and downloading the model
# from NMT.evaluate import calc_comet22
# comet_score, _ = calc_comet22(sources, predictions, references)
# print(f"COMET-22: {comet_score:.4f}")
```

## Common Issues and Solutions

### FastAlign Not Found
**Error**: `fast_align: command not found`

**Solution**:
```bash
# Verify FastAlign is installed
which fast_align

# If not found, install and add to PATH
# Follow: https://github.com/clab/fast_align
export PATH=/path/to/fast_align/build:$PATH
```

### Config File Path Issues
**Error**: Config parameter shows `$BASE_DIR` literally instead of expanding

**Solution**: Shell config files (.cfg) support variable expansion, but YAML files (.yaml) do not. Replace all `$BASE_DIR` with actual absolute paths in YAML configs:
```yaml
# Wrong
save: $BASE_DIR/models/nmt_models/an-en/PRETRAIN

# Correct
save: /home/username/charlotte-project/models/nmt_models/an-en/PRETRAIN
```

### SC Model ID Mismatch
**Error**: NMT training can't find SC-normalized data files

**Solution**: Ensure `sc_model_id` matches across configs:
- SC config: `SC_MODEL_ID=es2an-RNN-0`
- After pred_SC.sh, files have: `SC_es2an-RNN-0-RNN-0_es2an` (note the extra `-RNN-0`)
- Tokenizer config: `SC_MODEL_ID=es2an-RNN-0-RNN-0` (use the full ID from filenames)
- NMT config: `sc_model_id: es2an-RNN-0-RNN-0` (same as tokenizer)

### CUDA Out of Memory
**Error**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in NMT config:
```yaml
train_batch_size: 16  # Reduce from 32
val_batch_size: 16
test_batch_size: 16
```

### Empty Cognate List
**Error**: No cognates found after FastAlign

**Solution**: Adjust cognate threshold in SC config:
```bash
COGNATE_THRESH=0.6  # Increase from 0.5 to be more lenient
```

Or verify your parallel data actually contains related language pairs.

### Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'transformers'`

**Solution**: Ensure you've installed the correct requirements file:
```bash
# For SC and NMT training
pip install -r sound.requirements.txt

# For CopperMT
pip install -r copper.requirements.txt
```

## Typical Experimental Scenarios

**Baseline (No SC)**: Train NMT using only low-resource parallel data
- Skip Steps 1-3 (no SC training/application)
- Train tokenizer on low-resource data only (set `SC_MODEL_ID=null` in tokenizer config)
- Train NMT baseline (set `sc_model_id: null` in NMT config)

**SC Augmentation**: Use SC model to augment training with normalized high-resource data
- Complete all steps 1-9
- SC model transforms high-resource language to match low-resource orthography
- NMT benefits from increased vocabulary overlap

**Pretraining + Fine-tuning**: Pretrain on augmented high-resource data, then fine-tune on low-resource
- Pretraining: Set `from_pretrained: null` in config, use SC-augmented data
- Fine-tuning: Create new config with `from_pretrained: /path/to/pretrained/model/directory`
- Fine-tuning typically uses lower learning rate (1e-05) and fewer steps (10000)

# Pipeline
The code for running the main experiments is in the Pipeline directory.

The main Pipeline scripts are in the Pipeline directory. Skip to the documentation for each as needed:
- [Pipeline/train_SC.sh](#pipelinetrain_scsh) - For training (and scoring) an SC model.
- [Pipeline/pred_SC.sh](#pipelinepred_scsh) - For running inference with an SC model.
- [Pipeline/train_srctgt_tokenizer.sh](#pipelinetrain_srctgt_tokenizersh) - For training an NMT tokenizer.

[SC Configs](#sc-configs) are the backbone of the pipeline. They are used both by *Pipeline/train_SC.sh* and *Pipeline/pred_SC.sh*. An overview of SC Configs is given first, followed by documentation of the main Pipeline scripts. *Pipeline/train_srctgt_tokenizer.sh* utilizes its own config, which will be described in its own section of this documentation.

## SC Configs

See *Pipeline/cfg/SC* for the .cfg files for all 10 scenarios of these experiments. They contain the following parameters. If ever not using one of these parameters, as relvant (most should be used), then set it to null. See Pipeline/cfg/SC for details.
- **MODULE_HOME_DIR:** the system path to the *code* folder of this module, depending on where you cloned it on your system, e.g. *~/path/to/Cognate/code*
- **NMT_SRC:** source language in the low-resource (LR) direction we ultimately want to translate. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **NMT_TGT:** target language in the low-resource (LR) direction we ultimately want to translate. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **AUG_SRC:** source language of the high-resource (HR) direction we want to levarage. Should be a high-resource (HR) language closely related to *NMT_SRC*. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **AUG_TGT:** target language of the high-resource direction we want to leverage. Should be **THE SAME AS** *NMT_TGT*. Used by *make_nmt_configs.py* to make NMT config .yaml files. Not used by train_SC.sh or pred_SC.sh or tokenizer training scripts.
- **SRC:** the source language of the cognate prediction model. This should be the same as *AUG_SRC*. The goal is to use the resulting cognate prediction model to make *AUG_SRC* look more like *NMT_SRC* based on character correspondences.
- **TGT:** the target language of the cognate prediction model. This should be the same as *NMT_SRC*. The goal is to use the resulting cognate prediction model to make *AUG_SRC* look more like *NMT_SRC* based on character correspondences.
- **SEED:** a random seed used in different scripts, such as for randomizing data order
- **PARALLEL_(TRAIN|VAL|TEST):** Parallel train / val / test data .csv files. These are the parallel data used to train NMT models, and from which congates will be extracted to train the cognate prediction model.
- **APPLY_TO:** list (comma-delimited, no space) of more data .csv files to apply the cognate prediction model to. Not used by *train_SC.sh* but by *pred_SC.sh*.
- **NO_GROUPING:** Keep this set to True. Not sure I'll actually experiment with this. It's used when extracting the cognate list from the Fast Align results. Basically, if False, then "grouping" is applied. Don't worry about it. Ask Brendan if you really want to know.
- **SC_MODEL_TYPE:** 'RNN' or 'SMT'. Determines what kind of model will be trained to predict cognates.
- **COGNATE_TRAIN:** Directory where Fast Align results and cognate word lists are written. The final training data, however, will be created in *COPPERMT_DATA_DIR*. Don't ask why. It's inefficient copying of data in multiple places and I don't want to fix it at this point.
- **COGNATE_THRESH:** the normalized edit distance threshold to determine cognates. Parallel translation data is given to FastAlign which creats word pairs. Words pairs where the normalized edit distance is less than or equal to *COGNATE_THRESH* are considered cognates.
- **COPPERMT_DATA_DIR:** Directory where the cognate training data, model checkpoints, and predictions for each scenario will be saved. Each scenario will have its own subdirectory in this directory called *{SRC}_{TGT}_{SMT_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S={SEED}*, *e.g.*, *fr_mfe_RNN-0_S-0*.
- **COPPERMT_DIR:** The directory where the CopperMT repo was cloned, *e.g*, */home/hatch5o6/Cognate/code/CopperMT/CopperMT*.
- **PARAMETERS_DIR:** A folder to save the CopperMT parameters files
- **RNN_HYPERPARAMS:** A folder containing RNN hyperparameter files (each containing a hyperparameter set) and a *manifest.json* file mapping an id to each hyperparameter set (file) (RNNs only).
- **RNN_HYPERPARAMS_ID:** The RNN hyperparameter set (see *RNN_HYPERPARAMS*) to use to train an RNN model (RNNs only).
- **BEAM:** The number of beams used in beam-search decoding (RNNs only).
- **NBEST:** The number of hypotheses to generate. This should just be 1 (Not sure why it's even parameterized). (RNNs only).
- **REVERSE_SRC_TGT_COGNATES:** Will prepare data to be passed into FastAlign in format `{target language sentence} ||| {source language sentence}`, rather than in format `{source language sentence} ||| {target language sentence}`. It will result in slightly different data, but likely will not affect results much.
- **SC_MODEL_ID:** an ID given to the resulting cognate prediction model. This ID is used in other pipelines. Not used by train_SC.sh, but is used by pred_SC.sh to label the resulting noramlized high-resource (norm HR) file (the file that has replaced all words in the HR file with the respective predicted cognate).
- **ADDITIONAL_TRAIN_COGNATES_(SRC|TGT):** Parallel cognate files if wanting to add data from other sources, such as CogNet or EtymDB, to the training data. If not using, set to 'null'
- **(VAL|TEST)_COGNATES_(SRC|TGT):** Set these to the validation/test src/tgt files. If not passed, you should set *COGNATE_(TRAIN|VAL|TEST)_RATIO* to make train / val / test splits instead. If not using, set to 'null'. Should use either this or *COGNATE_(TRAIN|VAL|TEST)_RATIO*.
- **COGNATE_(TRAIN|VAL|TEST)_RATIO:** If not passing *(VAL|TEST)_COGNATES_(SRC|TGT)*, then these are the train / val / test ratios for splitting the cognate data. The three should add to 1. If not using, set to 'null'. Should use either this or *(VAL|TEST)_COGNATES_(SRC|TGT)*.


## Pipeline/train_SC.sh
This documentation is designed to walk you through the *Pipeline/train_SC.sh* script. You should read this documentation and the *train_SC.sh* script together. This documentation will refer to sections of the *train_SC.sh* code with numbers like 2.2 and 2.3.1.

**Pipeline/train_SC.sh** trains the character correspondence (SC) models.
We call it SC, which stands for "sound correspondence", but more accurately, what we're more acurately detecting are actually character correspondences, since we apply this on orthography rather than phones.

**Pipeline/train_SC.sh** is run from /Cognate/code, and takes a single positional argument, one of the *.cfg* config files described [above](#sc-configs), e.g.:
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
It uses these parameters from the SC Config file: 
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

This script will write the result to *{COGNATE_TRAIN}/fastalign/{SRC}-{TGT}.txt*, which writes each sentence pair to a line in the format ```{source language sentence} ||| {target language sentence}```. 

If *REVERSE_SRC_TGT_COGNATES* is set to *true*, then the source and target sentences will be flipped: ```{target language sentence} ||| {source language sentence}```. This setting will result in slightly different cognate training data, but should likely not have significant impact on results. Should probably just keep *REVERSE_SRC_TGT_COGNATES* set to *false*.

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

If *REVERSE_SRC_TGT_COGNATES* is *true*, then *TGT* will be passed as for *--src* and *SRC* will be passed for --*tgt*, just because we flipped the source and target sentences when running *word_alignments/prepare_for_fastalign.py* in **2.3.2**.

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

**CopperMT/assert_no_overlap_in_formatted_data.py**
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
#UPDATE I don't think these paths are being used for anything anymore.

#### 4.2 Run inference on the test set
To calculate scores, inference is first run on the test set.

**Inference with an RNN model**  
To run inference with an RNN model, *{COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan_PREDICT.sh* is called, passing the Copper MT parameters file from **3.2.6** (*PARAMETERS_F*), the path to the selected RNN checkpoint from **3.2.7** (*SELECTED_RNN_CHECKPOING*), *SEED*, an indicator "test", *NBEST*, and *BEAM*. This script will save its results to a file whose path is saved to the variable *HYP_OUT_TXT*. This path should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/results/test_selected_checkpoint_{SRC}_{TGT}.{TGT}/generate-test.txt*.

The model hypotheses need to be extracted from the *HYP_OUT_TXT* file, which is done with the **NMT/hr_CopperMT.py** script. This script has three modes: "prepare", "retrieve", and "get_test_results". Modes "prepare" and "retrieve" will be discussed later in connection to *pred_SC.sh* [below](#24-apply-sc) To extract the hypotheses from the model test results file, we use mode "get_test_results". Only the parameters relevant to this mode are shown here. This mode will write the hypotheses to a file parallel to the source file, where on each line is simply the cognate hypothesis for each source word.

**NMT/hr_CopperMT.py (get_test_results)**
* *--function, -F:* The script mode. In this case, it should be "get_test_results".
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



## Pipeline/pred_SC.sh
This documentation is designed to walk you through the *Pipeline/pred_SC.sh* script. You should read this documentation and the *pred_SC.sh* script together. This documentation will refer to sections of the *pred_SC.sh* code with numbers like 2.2 and 2.3.

**Pipeline/pred_SC.sh** runs inference of an SC model. It is run from /Cognate/code, and takes a single positional argument, one of the *.cfg* config files described [above](#sc-configs), e.g.:
```
bash Pipeline/pred_SC.sh /home/hatch5o6/Cognate/code/Pipeline/cfg/SC/fr-mfe.cfg
```

For each parallel data .csv file in *PARALLEL_(TRAIN|VAL|TEST)* and *APPLY_TO*, this script looks for any source or target sentence file in the *.csv* for the language *SRC* -- that is, even if there is a target file in the *.csv* for the language *SRC* it will be included -- and applies the SC model to *each* word of *each* sentence and then saves the result to a new file. We then have data in the *SRC* language that is made more similary to the *TGT* language based on learned character correspondences.

### 1) ARGUMENTS
It uses these parameters from the SC Config file:
- MODULE_HOME_DIR
- SRC
- TGT
- PARALLEL_TRAIN
- PARALLEL_VAL
- PARALLEL_TEST
- APPLY_TO
- SC_MODEL_TYPE
- SEED
- SC_MODEL_ID
- COPPERMT_DATA_DIR
- COPPERMT_DIR
- PARAMETERS_DIR
- RNN_HYPERPARAMS_ID
- BEAM
- NBEST

### 2) APPLY SC MODEL
#### 2.0 Alter SC_MODEL_ID
We simply append the *SC_MODEL_TYPE* ('SMT' or 'RNN') and the *RNN_HYPERPARAMS_ID* to the end of *SC_MODEL_ID*, just so we can track precisely which version of a model was used to apply character correspondence later on.

#### 2.1 Write the CopperMT parameters file
First the CopperMT parameters file is written. This step is just like step **3.2.6** under [Pipeline/train_SC.sh](#pipelinetrain_scsh). 

The parameters file required by the CopperMT module needs to be written, which is performed by *Pipeline/write_scripts.py*.

**Pipeline/write_scripts.py**
* *--src:* Source language code. 
* *--tgt:* Target language code
* *--coppermt_data_dir:* Parent folder containing the training data, models, and outputs of each cognate prediction scenario. Should be *COPPERMT_DATA_DIR*.
* *--sc_model_type:* The SC model type, either "RNN" or "SMT". Should be *SC_MODEL_TYPE*.
* *--rnn_hyperparams_id:* The id corresonding to the desired RNN hyperparams set. Should be *RNN_HYPERPARAMS_ID*.
* *--seed:* Should be *SEED*.
* *--parameters, -p:* The path the CopperMT parameters file will be written to. Should be *{PARAMETERS_DIR}/parameters.{SRC}-{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}.cfg*

#### 2.2 Get selected SC model
If running an RNN model, we need to retrieve the path to the best model. This should be *COPPERMT_DATA_DIR/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/checkpoints/selected.pt*, which is set to the variable *SELECTED_RNN_CHECKPOINT*.

#### 2.3 Delete inference directories if pre-existing
This is the same as **4.1** under [Pipeline/train_SC.sh](#pipelinetrain_scsh).

A couple directories, if pre-existing, are deleted.
#TODO describe what they are after you figure this out. (Are they still used or did we change this?? The files aren't called elsewhere in train_SC.sh or pred_SC.sh). Run the script, and see afterwards if they exist.
#UPDATE I don't think these paths are being used for anything anymore.

#### 2.4 APPLY SC
The SC model is then applied to text files corresponding to the language *SRC*, making the data look more like language *TGT* based on character correspondences.

To do this, words from the text files need to be prepared for the SC model. This is done with the *NMT/hr_CopperMT.py* script run in the "prepare" mode on each *.csv* file in *PARALLEL_(TRAIN|VAL|TEST)* and in the comma-delimited list *APPLY_TO*. In each *.csv* are the source and target parallel text files. The script grabs all text files corresponding to the language *SRC*, regardless of whether they are set as a source or target file in the *.csv*, and from them compiles a list of unique words. The arguments applicable to the "prepare" mode are described here.

**NMT/hr_CopperMT.py (prepare)**
* *--function, -F:* To run in "prepare" mode, set this to "prepare". "prepare" is also the default value, so if this parameter is not specified, it will run in "prepare" mode.
* *--data:* The path to a parallel data *.csv* file.
* *--out:* The directory in which will be written the list of unique words extracted from the text files listed in the *--data .csv* file.
* *--hr_lang, -hr:* The high-resource language. Should be *SRC*.
* *--lr_lang, -lr:* The low-resource language. Should be *TGT*.
* *--training_data:* The folder where the SC model training data was written. Should be *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{SC_MODEL_TYPE}-{RNN_HYPERPARAMS_ID}_S-{SEED}/inputs/split_data/{SRC}_{TGT}/{SEED}*.
* *--limit_lang:* The language whose text files we want to grab from the *.csv*. Should be *SRC*.

Afterwards, we can run inference.

**Inference with RNN model**
If infering with an RNN model, we run *{COPPERMT_DIR}/pipeline/main_nmt_bilingual_full_brendan_PREDICT.sh*, passing in the CopperMT parameters file (*PARAMETERS_F*) from **2.1**, the path to the best checkpoint (*SELECTED_RNN_CHECKPOINT*) from **2.2**, *SEED*, the tag "inference", *NBEST*, and *BEAM*.

This will predict the cognates for each of the words in our list created by *hr_CopperMT.py (prepare)*. Its output will be saved to the path *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_RNN-{RNN_HYPERPARAMS_ID}_S-{SEED}/workspace/reference_models/bilingual/rnn_{SRC}-{TGT}/{SEED}/results/inference_selected_checkpoint_{SRC}_{TGT}.{TGT}/generate-test.txt*, which is saved to variable *COPPERMT_RESULTS*. Then we run NMT/hr_CopperMT.py in "retrieve" mode, which for each word in the high resource text files, it replaces it with its predicted cognate.

**NMT/hr_CopperMT.py (retreive), for RNN**
* *--function:* Set to 'retrieve'.
* *--data:* The path to the parallel data .csv file, listing the text files where we want to replace each word with its predicted cognate. Should be the same file we pass in for *--data* in the 'prepare' model.
* *--CopperMT_results:* The output from the RNN model. Should be path saved to *COPPERMT_RESULTS* described above.
* *--hr_lang, -hr:* The high-resource language. Should be *SRC*. For each parallel text file (in the .csv passed as *--data*) corresponding to this language, each word in the file will be replaced with its predicted cognate. Should be *SRC*.
* *--lr_lang, -lr:* The low-resource language. Should be *TGT*.
* *--MODEL_ID:* Set this to *SC_MODEL_ID*. A copy of the *SRC* text files will be saved to the original path but with the string "SC_{SC_MODEL_ID}_{SRC}2{TGT}" inserted into the file name just before the file extension, indicating it is the version of the data where words have been replaced with their predicted cognates. For example, if predicting cognates for the text file "source.txt", the results will be saved to "source.SC_{SC_MODEL_ID}_{SRC}2{TGT}.txt". This file is the final result of the cognate prediction where each word has been replaced with a predicted cognate.


**Inference with SMT model**
If inferring with an SMT model, we run *{COPPERMT_DIR}/pipeline/main_smt_full_brendan_PREDICT.sh*, passing in the CopperMT parameters file (*PARAMETERS_F*) from **2.1**, the file path of the source sentences (*TEXT*), a template for the outputs (*HYP_OUT*), and *SEED*. This functions similarily to inference with the RNN model, where it's predicting cognates for each word in the list created by *hr_CopperMT.py (prepare)*. The outputs are written to *{COPPERMT_DATA_DIR}/{SRC}_{TGT}_SMT-null_S-{SEED}/inputs/split_data/{SRC}_{TGT}/inference/test_{SRC}_{TGT}.{TGT}.hyp.txt*, which is saved to variable *HYP_OUT_F*. We then run *NMT/hr_CopperMT.py* in "retrieve" mode, which for each word in the high resource text files, it replaces it with its predicted cognate.

**NMT/hr_CopperMT.py (retrieve), for SMT**
The parameters passed in "retrieve" mode for SMT model results are exactly the same as those for RNN model results **EXCEPT** instead of **--CopperMT_results**, we use the parameter **--CopperMT_SMT_results**, which will be set to the output of the SMT model, saved to the variable *HYP_OUT_F* described above.

We are then done! Hurray! :D


## Pipeline/train_srctgt_tokenizer.sh
This is documentation for the *Pipeline/train_srctgt_tokenizer.sh* script, which is used to train an SentencePiece (https://github.com/google/sentencepiece) tokenizer. This scripts requires a Tok Config .cfg file and trains a single tokenizer for all provided source and target languages. The Tok Config file contains the following fields:

See the config files in *Pipeline/cfg/tok_NO* for examples.

- **SPM_TRAIN_SIZE:** This is the total number of lines of data to use to train the tokenizer. Provided data will be down- / upsampled to this number.
- **SRC_LANGS:** A comma-delimitted list of source language codes (no spaces). *E.g.* 'en', 'en,fr', 'en,fr,it'.
- **SRC_TOK_NAME:** A source name for the tokenizer. I like to use a hyphen-delimited list of the source langs. *E.g.* 'en-fr'. The tokenizer name will be *{SRC_TOK_NAME}_{TGT_TOK_NAME}*.
- **TGT_LANGS:** A comma-delimitted list of target language codes (no spaces). *E.g.* 'es', 'es,pt', 'es,pt,en'.
- **TGT_TOK_NAME:** A target name for the tokenizer. I like to use a hyphen-delimited list of the target langs. The tokenizer name will be *{SRC_TOK_NAME}_{TGT_TOK_NAME}*.
- **DIST:** A string representing what percentage of the training data is to be assigned to each source/target language. Percentages are assigned in the format "{language code}:{percentage}" in a comma-delimited list (no spaces). For example, given "bn:25,as:25,hi:50", the *bn* language data will be down- / upsampled until it is 25% of the *SPM_TRAIN_SIZE*, *as* until it is 25%, and *hi* until it is 50%. The percentages in the string must add up to 100.
- **(TRAIN|VAL|TEST)_PARALLEL:** These are comma-delimited lists pointing to parallel data *.csv* files (the same used for training cognate-prediction models). The parallel data in these files will be used as tokenizer training data in this script. Note that there is no real functional distinction here between *TRAIN_PARALLEL* from *VAL_PARALLEL* and *TEST_PARALLEL*, as all files will be used to gather training data. They are just used for organizational purposes.
- **TOK_TRAIN_DATA_DIR:** The folder where the tokenizer training data and models will be written to.
- **SC_MODEL_ID:** If relevant (*e.g.*, if including SC parallel data *.csv* files, such as from *NMT/data/SC*), then this is SC_MODEL_ID of the cognate prediction model that was used to alter the high-resource data. This is used to read the right versions of the parallel text files. If not relevant, set this to "null".
- **VOCAB_SIZE:** The voacbulary size of the model.
- **SPLIT_ON_WS:** If "true", then a whitespace token "_" will be added to the sentencepiece module and compell segmentation on whitespace. If "false", then no whitespace token is created.
- **INCLUDE_LANG_TOKS:** If "true", special language tokens for the provided *SRC_LANGS* and *TGT_LANGS* will be added to the model.
- **INCLUDE_PAD_TOK:** If "true", will include a padding token ("<pad>") in the tokenizer.
- **SPECIAL_TOKS:** A comma-delimited list of other special tokens you want to add to the tokenizer (no spaces between elements, *e.g. SPECIAL_TOKS=<these>,<are>,<special>,<tokens>*). Set to "null" to not pass in any additional special tokens.
- **IS_ATT:** If this the tokenizer will be used in an experiment where sound correspondences are applied to the target language to create parallel pretraining data (such as for the *en2djk-djk_en* or *hi2bho-bho_hi* scenarios), then set to "true". Otherwise, set to "false".

The script will read all the parallel data *.csv* files provided and extract the data corresponding to the provided *SRC_LANGS* and *TGT_LANGS*. The distinction between *TRAIN*, *VAL* and *TEST* parallel data does not matter, except for organizational purposes. All of it will be gathered. The entirety of this data will be written to files corresponding to each language inside of *TOK_TRAIN_DATA_DIR* (*e.g., en.txt, fr.txt, mfe.txt*). These files will be read to create the final collection of tokenizer training data, which will contain *SPM_TRAIN_SIZE* lines.


A subfolder called *{SRC_TOK_NAME}_{TGT_TOK_NAME}* will be created inside *TOK_TRAIN_DATA_DIR*. Inside *{SRC_TOK_NAME}_{TGT_TOK_NAME}* will be written the following files:
    - *data_dict.json:* a dictionary where the keys are the data files used to train the tokenizer. These will point to the same language data files in *TOK_TRAIN_DATA_DIR*. The values are the fraction of the total tokenizer training data (*SPM_TRAIN_SIZE*) that will come from the respective file. The data in each file will be up- or down-sampled to meet this ammount.
    - *{SRC_TOK_NAME}_{TGT_TOK_NAME}.model:* The spm model
    - *{SRC_TOK_NAME}_{TGT_TOK_NAME}.vocab:* The spm vocabulary file
    - *training_data.s=1500.txt:* The final collection of tokenizer training data extracted from the parallel data *.csv* files. This will contain *SPM_TRAIN_SIZE* number of sentences with the per language distribution specified in *DIST*.
    - *training_data.s=1500div={language code}.txt:* For each language in *SRC_LANGS* and *TGT_LANGS*, a file containing the subset of final tokenizer data in *training_data.s=1500.txt* pertaining to the language. These files are not used for anything except as a way of logging the per-language training data. Only *training_data.s=1500.txt* is read by the SentencePiece trainer.

# NMT Training

The NMT training system is located in the *NMT/* directory and uses PyTorch Lightning with BART (Bidirectional and Auto-Regressive Transformers) models for sequence-to-sequence translation.

## NMT/train.py

The main training script *NMT/train.py* supports three modes:

### Running Modes

**TRAIN Mode**: Train a new model or fine-tune an existing one
```bash
python NMT/train.py -c configs/your-config.yaml -m TRAIN
```

**TEST Mode**: Evaluate a trained model on test data
```bash
python NMT/train.py -c configs/your-config.yaml -m TEST
```

**INFERENCE Mode**: Run predictions on new data
```bash
python NMT/train.py -c configs/your-config.yaml -m INFERENCE
```

## NMT Configuration Files

NMT configs are YAML files located in *NMT/configs/*. See *NMT/configs/CONFIGS/* for examples organized by language pair. Each config file contains the following parameter groups:

### Output Parameters
- **src**: Source language code
- **tgt**: Target language code
- **save**: Directory where model checkpoints, logs, and predictions will be saved. A subdirectory *{save}_TRIAL_s={seed}* will be created.
- **test_checkpoint**: Path to checkpoint for testing/inference. Set to `null` to auto-select best checkpoint based on validation loss.
- **remove_special_toks**: (Boolean) Whether to remove special tokens (BOS/EOS/PAD) from generated predictions
- **verbose**: (Boolean) Print detailed training batch information every 500 batches (for debugging)
- **little_verbose**: (Boolean) Print training batch information every 10,000 batches

### Fine-tuning Parameters
- **from_pretrained**: Path to a pretrained model checkpoint to fine-tune from. Set to `null` to train from scratch.
  - Can be a directory (will auto-select best checkpoint) or a specific `.ckpt` file

### Data Parameters
- **train_data**: Path to training data CSV file. Must end with `/train.no_overlap_v1.csv`
- **val_data**: Path to validation data CSV file. Must end with `/val.no_overlap_v1.csv`
- **test_data**: Path to test data CSV file. Must end with `/test.csv`
- **append_src_token**: (Boolean) Prepend `<src_lang>` token to source sentences
- **append_tgt_token**: (Boolean) Prepend `<tgt_lang>` token to target sentences
- **upsample**: (Boolean) Upsample smaller language pairs to match largest pair (training only)
- **sc_model_id**: ID of the SC model used to normalize data. Set to `null` if not using SC-normalized data. Used to locate correct data files with SC model applied.

### Tokenizer Parameters
- **spm**: Path to the SentencePiece model (without extension). Should point to files created by *Pipeline/train_srctgt_tokenizer.sh*
- **do_char**: (Boolean) Use character-level tokenization instead of SentencePiece. If `true`, vocabulary will be built from training data.

### Training Parameters
- **n_gpus**: Number of GPUs to use. Uses DDP strategy if ≥ 1
- **seed**: Random seed for reproducibility
- **qos**: (Optional) SLURM QOS parameter for cluster scheduling
- **max_steps**: Maximum training steps
- **train_batch_size**: Training batch size
- **val_batch_size**: Validation batch size
- **test_batch_size**: Test batch size
- **early_stop**: Early stopping patience (number of validation checks without improvement)
- **save_top_k**: Number of best checkpoints to keep based on validation loss
- **val_interval**: Validation frequency as fraction of epoch (e.g., 0.5 = validate twice per epoch)
- **learning_rate**: Learning rate (e.g., 2e-04, 5e-04)
- **weight_decay**: Weight decay for AdamW optimizer
- **device**: Device for training (`cuda` or `cpu`)

### Model Architecture Parameters

**Encoder Configuration:**
- **encoder_layers**: Number of encoder layers (typically 6)
- **encoder_attention_heads**: Number of attention heads in encoder (typically 8)
- **encoder_ffn_dim**: Feed-forward network dimension in encoder (typically 2048)
- **encoder_layerdrop**: Encoder layer dropout probability (0.0 to disable)

**Decoder Configuration:**
- **decoder_layers**: Number of decoder layers (typically 6)
- **decoder_attention_heads**: Number of attention heads in decoder (typically 8)
- **decoder_ffn_dim**: Feed-forward network dimension in decoder (typically 2048)
- **decoder_layerdrop**: Decoder layer dropout probability (0.0 to disable)

**General Model Configuration:**
- **max_position_embeddings**: Maximum sequence length (typically 512)
- **max_length**: Maximum generation length for predictions (typically 512)
- **d_model**: Model dimension / hidden size (typically 512)
- **dropout**: Dropout probability (typically 0.1)
- **activation_function**: Activation function (typically `gelu`)

## Training Output Structure

When training, the following directory structure is created under *{save}_TRIAL_s={seed}*:

```
{save}_TRIAL_s={seed}/
├── checkpoints/              # Model checkpoints
│   └── epoch=X-step=Y-val_loss=Z.ckpt
├── logs/                     # Training logs (CSV format)
│   └── version_0/
│       └── metrics.csv
└── predictions/              # Test predictions and metrics
    └── {checkpoint_name}/
        ├── predictions.txt
        └── metrics.json      # BLEU and chrF scores
```

### Training Metrics and Loss Curves

Training and validation metrics are **automatically logged** during training to `{save}_TRIAL_s={seed}/logs/version_0/metrics.csv`.

The CSV file contains columns including:
- `train_loss_step`: Training loss per step
- `train_loss_epoch`: Training loss per epoch (aggregated)
- `val_loss`: Validation loss
- `epoch`: Current epoch
- `step`: Current training step
- `lr-AdamW`: Learning rate (tracked by LearningRateMonitor)

**Visualizing Loss Curves**:

You can plot training and validation loss curves using Python:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
metrics = pd.read_csv('path/to/{save}_TRIAL_s={seed}/logs/version_0/metrics.csv')

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(metrics['step'], metrics['train_loss_step'], label='Train Loss', alpha=0.6)
plt.plot(metrics['step'], metrics['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curves.png')
plt.show()
```

**Alternative: TensorBoard**

The PyTorch Lightning logger also supports TensorBoard. To view logs with TensorBoard:

```bash
tensorboard --logdir {save}_TRIAL_s={seed}/logs
```

Then open your browser to `http://localhost:6006` to view interactive loss curves and other metrics.

## Evaluation Metrics

The testing mode (*TEST*) evaluates models using:
- **BLEU**: Bilingual Evaluation Understudy score (via sacrebleu)
- **chrF**: Character n-gram F-score (via sacrebleu)

These metrics are automatically computed during TEST mode and saved to `metrics.json` in each checkpoint's prediction directory.

If multiple checkpoints exist, all will be tested and the best BLEU score will be reported in *predictions/all_scores.json*.

### Optional: COMET Evaluation

**COMET** (Crosslingual Optimized Metric for Evaluation of Translation) is a neural learned metric that correlates better with human judgments than BLEU/chrF. The code includes COMET-22 support in `NMT/evaluate.py`, but it requires additional setup:

**Installing COMET**:
```bash
pip install unbabel-comet
```

**Downloading COMET Model**:
```bash
# Download wmt22-comet-da model
comet-download --model Unbabel/wmt22-comet-da

# Note the download location (typically ~/.cache/comet/)
```

**Computing COMET Scores Manually**:

The current TEST mode only computes BLEU and chrF automatically. To compute COMET scores on your test results:

```python
from NMT.evaluate import calc_comet22

# Read your data
with open('predictions/checkpoint-name/predictions.txt') as f:
    hypotheses = [line.strip() for line in f]
with open('data/test.src') as f:
    sources = [line.strip() for line in f]
with open('data/test.tgt') as f:
    references = [line.strip() for line in f]

# Compute COMET (requires GPU)
system_score, sentence_scores = calc_comet22(sources, hypotheses, references)
print(f"COMET-22: {system_score:.4f}")
```

**Important**: The COMET model path in `NMT/evaluate.py:52` is hardcoded:
```python
comet22_path = "/home/hatch5o6/nobackup/archive/comet/wmt22-comet-da/checkpoints/model.ckpt"
```

Before using COMET, you must update this line to point to your downloaded model location. Find the model path with:
```bash
python -c "from comet import download_model; print(download_model('Unbabel/wmt22-comet-da'))"
```

**Why COMET is Optional**:
- Requires additional model download (~2GB)
- Requires GPU for reasonable speed
- Not computed automatically during TEST mode
- BLEU and chrF are sufficient for most research comparisons

## Example Configurations

**Baseline NMT (no SC augmentation)**:
```yaml
# Example: NMT/configs/CONFIGS/an-en/NMT.an-en.yaml
src: an
tgt: en
save: /path/to/output/an-en/NMT.an-en
from_pretrained: null
train_data: /path/to/data/PLAIN/an-en/train.no_overlap_v1.csv
spm: /path/to/spm_models/es-an_en/es-an_en/es-an_en
sc_model_id: null
max_steps: 20000
learning_rate: 2e-04
```

**SC-Augmented Pretraining**:
```yaml
# Example: NMT/configs/CONFIGS/mfe-en/PRETRAIN.SC_fr2mfe-en.yaml
src: fr
tgt: en
save: /path/to/output/mfe-en/PRETRAIN.SC_fr2mfe-en
from_pretrained: null
train_data: /path/to/data/SC/SC_fr2mfe-en/train.no_overlap_v1.csv
spm: /path/to/spm_models/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en/SC_fr2mfe-mfe_en
sc_model_id: FR-MFE-RNN-0-RNN-66
max_steps: 250000
learning_rate: 5e-04
```

**Fine-tuning from Pretrained**:
```yaml
# Example: Fine-tune on low-resource data
src: mfe
tgt: en
from_pretrained: /path/to/pretrained/model/directory
train_data: /path/to/data/PLAIN/mfe-en/train.no_overlap_v1.csv
max_steps: 20000
learning_rate: 2e-04
```

## Data Format

Training data CSV files follow the same format as SC model training (see [Parallel Data CSV Files](#parallel-data-csv-files)), with header:
```
src_lang,tgt_lang,src_path,tgt_path
```

The *MultilingualDataset* class in *NMT/parallel_datasets.py* handles:
- Loading parallel data from CSV specifications
- Language token prepending (if configured)
- Upsampling to balance language pairs (if configured)
- Filtering by language pair

## Advanced Features

**Multilingual Training**: The system supports training on multiple language pairs simultaneously by including multiple pairs in the CSV files. The *upsample* parameter can balance training across pairs.

**SC Model Integration**: When *sc_model_id* is set, the system looks for data files with the SC model ID in the filename, allowing seamless integration of SC-normalized data.

**Checkpoint Selection**: During testing, if no specific checkpoint is provided, the system evaluates all available checkpoints and selects the best based on BLEU score.

**Learning Rate Scheduling**: The system uses linear warmup (5% of max_steps) followed by linear decay with AdamW optimizer.
