# CharLOTTE Experimentation Guide

**Purpose:** Complete end-to-end workflow for running CharLOTTE experiments from SC model training to NMT evaluation.

---

**🎯 Who is this for?**
- Researchers who completed [QUICKSTART.md](QUICKSTART.md) and want **full manual control**
- Users preparing to run experiments on **their own language pairs**
- Anyone who needs to **understand and customize** each phase of the pipeline

**🎯 What you'll do:**
- Manually create configuration files for SC models, tokenizers, and NMT
- Execute each of the 6 phases step-by-step with full visibility
- Learn to adjust parameters and troubleshoot issues independently

**🎯 Prerequisites:**
- ✅ Completed [SETUP.md](SETUP.md) installation
- ✅ Passed the quickstart test (full 6-phase pipeline)
- ✅ **Recommended:** Run [QUICKSTART.md](QUICKSTART.md) first to understand the workflow

---

**New to CharLOTTE?** If you haven't run the quickstart yet, start with [QUICKSTART.md](QUICKSTART.md) for a 30-minute guided introduction that will get you first results with a working Spanish→Portuguese example.

---

## Table of Contents

1. [Overview](#overview)
2. [Before You Begin](#before-you-begin)
3. [End-to-End Example: Portuguese→English](#end-to-end-example-portugueseenglish)
4. [Next Steps](#next-steps)

---

## Overview

CharLOTTE (Character-Level Orthographic Transfer for Token Embeddings) is a framework for low-resource neural machine translation (NMT) that leverages sound correspondence (SC) models to augment training data.

### The CharLOTTE Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. PREPARE DATA                                                 │
│    • Obtain parallel corpora for low-resource and high-resource │
│    • Create train/val/test splits                               │
│    • Format as CSV metadata files                               │
│    └─> See: DATA_PREPARATION.md                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. TRAIN SC MODEL (Sound Correspondence)                        │
│    • Extract cognates from related language pairs               │
│    • Train RNN or SMT model for character transformations       │
│    • Evaluate on character-level BLEU                           │
│    └─> Configure: CONFIGURATION.md#sc-model-configuration       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. APPLY SC MODEL                                               │
│    • Transform high-resource data to look like low-resource     │
│    • Creates SC-normalized parallel data                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. TRAIN TOKENIZER                                              │
│    • Train SentencePiece on combined data                       │
│    • Balanced sampling across languages                         │
│    └─> Configure: CONFIGURATION.md#tokenizer-configuration      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. TRAIN NMT MODEL                                              │
│    • Train Transformer on low-resource + SC-augmented data      │
│    • Monitor with TensorBoard or CSV metrics                    │
│    └─> Configure: CONFIGURATION.md#nmt-configuration            │
│    └─> Monitor: MONITORING.md#monitoring-nmt-training           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. EVALUATE                                                     │
│    • Compute BLEU and chrF on test set                          │
│    • Compare with baseline (no SC augmentation)                 │
│    └─> Evaluate: MONITORING.md#evaluating-nmt-models            │
└─────────────────────────────────────────────────────────────────┘
```

### When to Use CharLOTTE

CharLOTTE is most effective when:

✅ **You have a low-resource language pair** with 5,000-50,000 parallel sentences
✅ **You have a related high-resource pair** with 100,000+ parallel sentences
✅ **Languages are related** (share linguistic ancestry or are in contact)
✅ **Target quality improvement** of 5-15 BLEU points over baseline

❌ **Don't use CharLOTTE if**:
- Languages are completely unrelated (no shared vocabulary patterns)
- Low-resource data < 2,000 pairs (insufficient for NMT)
- High-resource data < 50,000 pairs (SC model won't learn robust patterns)

**See [DATA_PREPARATION.md](DATA_PREPARATION.md)** for detailed guidance on data requirements and obtaining corpora.

---

## Before You Begin

### 1. Review Documentation Structure

This guide focuses on the **end-to-end workflow**. For detailed information on specific topics:

| Topic | Document | Use When |
|-------|----------|----------|
| **Installation** | [SETUP.md](SETUP.md) | Setting up CharLOTTE for the first time |
| **Quick Start** | [QUICKSTART.md](QUICKSTART.md) | Want to see results in 30 minutes |
| **Data Preparation** | [DATA_PREPARATION.md](DATA_PREPARATION.md) | Obtaining and formatting training data |
| **Configuration** | [CONFIGURATION.md](CONFIGURATION.md) | Understanding all config parameters |
| **Monitoring & Evaluation** | [MONITORING.md](MONITORING.md) | Tracking training and measuring quality |
| **Troubleshooting** | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Debugging errors and issues |

### 2. Verify Your Installation

Before proceeding, verify all components are installed:

```bash
# Check SCIMT installation
ls $SCIMT_DIR/Pipeline/train_SC_venv.sh  # Should exist

# Check virtual environments (all three required)
source $SCIMT_DIR/venv_sound/bin/activate && python -c "import torch; print('sound env OK')"
source $SCIMT_DIR/venv_copper/bin/activate && python -c "import fairseq; print('copper env OK')"
source $SCIMT_DIR/venv_nmt/bin/activate && python -c "import lightning; print('nmt env OK')"

# Check FastAlign
which fast_align  # Should print path to binary

# Check CopperMT installation
ls $SCIMT_DIR/CopperMT/CopperMT/submodules/mosesdecoder/bin/moses  # Should exist
```

If any checks fail, return to [SETUP.md](SETUP.md).

---

## End-to-End Example: Portuguese→English

This section demonstrates the complete CharLOTTE workflow using a realistic low-resource scenario.

### Scenario Overview

**Goal**: Train a Portuguese→English NMT system with limited Portuguese data by leveraging Spanish→English data.

**Language Pairs**:
- **Low-resource**: Portuguese (pt) → English (en) — 10,000 parallel sentences
- **High-resource**: Spanish (es) → English (en) — 300,000 parallel sentences

**Strategy**:
1. Train SC model to transform Spanish → Portuguese at character level
2. Apply SC model to Spanish-English data, creating Portuguese-like text
3. Train NMT on combined Portuguese + SC-normalized Spanish data

**Expected Results**:
- Baseline (10k pt-en only): BLEU ~18
- With SC augmentation: BLEU ~28 (+10 BLEU points)

### Understanding the Workflow Structure

This guide walks through CharLOTTE's **6-phase pipeline** with detailed implementation tasks. Each phase typically involves two tasks: (1) creating a configuration file, then (2) executing the phase.

**The 6 phases:**
1. **Prepare Data** *(prerequisite)* - Obtain corpora and create CSV files
2. **Train SC Model** - Learn character correspondences between languages
3. **Apply SC Model** - Transform high-resource data to augment low-resource training
4. **Train Tokenizer** - Build shared vocabulary across language pairs
5. **Train NMT Model** - Train translation model with augmented data
6. **Evaluate** - Measure translation quality and compare with baseline

**Navigation tip**: Each phase section below is self-contained with configuration templates and execution commands. Follow them sequentially for best results.

---

### Prerequisites: Prepare Your Data

Before starting the workflow, you need:

1. **Low-resource parallel data**: Portuguese-English train/val/test splits
2. **High-resource parallel data**: Spanish-English train/val/test splits
3. **CSV metadata files** pointing to your data

**If you don't have data yet**, see [DATA_PREPARATION.md](DATA_PREPARATION.md) for:
- Obtaining data from OPUS/Tatoeba
- Creating train/val/test splits with Python script
- Formatting as CSV files
- Recommended directory structure

**Assumed directory structure for this example**:
```
$BASE_DIR/
├── data/
│   ├── raw/
│   │   ├── low-resource/
│   │   │   ├── train.pt  (Portuguese source)
│   │   │   ├── train.en  (English target)
│   │   │   ├── val.pt, val.en
│   │   │   └── test.pt, test.en
│   │   └── high-resource/
│   │       ├── train.es  (Spanish source)
│   │       ├── train.en  (English target)
│   │       ├── val.es, val.en
│   │       └── test.es, test.en
│   └── csv/
│       ├── train.no_overlap_v1.csv
│       ├── val.no_overlap_v1.csv
│       └── test.csv
├── models/
│   ├── sc_models/
│   ├── tokenizers/
│   └── nmt_models/
└── configs/
    ├── sc/
    ├── tok/
    └── nmt/
```

### Phase 1: Prepare Data - Setup Project Directories

```bash
# Create directory structure
mkdir -p $BASE_DIR/{data/{raw/{low-resource,high-resource},csv},models/{sc_models,tokenizers,nmt_models},configs/{sc,tok,nmt}}

# Verify
ls -la $BASE_DIR
```

---
**Progress**: ✅ Data Ready | 🔄 Train SC Model | ⬜ Apply SC Model | ⬜ Train Tokenizer | ⬜ Train NMT | ⬜ Evaluate
---

### Phase 2: Train SC Model

#### Task 2.1: Create SC Model Configuration

Create `$BASE_DIR/configs/sc/es2pt.cfg`:

```bash
# SC Model Configuration: Spanish → Portuguese
# This trains a character-level transformation model

# Paths
MODULE_HOME_DIR=$SCIMT_DIR
COPPERMT_DIR=$SCIMT_DIR/CopperMT/CopperMT

# Language pair for SC model
SRC=es    # Spanish (high-resource source)
TGT=pt    # Portuguese (low-resource target)
SEED=1000

# Data (CSV files containing parallel data paths)
PARALLEL_TRAIN=$BASE_DIR/data/csv/train.no_overlap_v1.csv
PARALLEL_VAL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
PARALLEL_TEST=$BASE_DIR/data/csv/test.csv

# Where to apply the trained SC model
APPLY_TO=$PARALLEL_TRAIN,$PARALLEL_VAL,$PARALLEL_TEST

# SC model settings
SC_MODEL_TYPE=RNN             # Options: RNN (neural) or SMT (phrase-based)
COGNATE_THRESH=0.6            # Similarity threshold: 0.5 (strict) to 0.7 (lenient)
NO_GROUPING=true              # Recommended: disable grouping

# Output directories
SC_MODEL_ID=es2pt-RNN-0
COGNATE_TRAIN=$BASE_DIR/models/sc_models/cognates
COPPERMT_DATA_DIR=$BASE_DIR/models/sc_models
PARAMETERS_DIR=$BASE_DIR/configs/sc/parameters

# RNN-specific settings
RNN_HYPERPARAMS=$MODULE_HOME_DIR/Pipeline/parameters/rnn_hyperparams
RNN_HYPERPARAMS_ID=0          # 0 = default hyperparameters
BEAM=5
NBEST=1

# Cognate extraction
REVERSE_SRC_TGT_COGNATES=false
COGNATE_TRAIN_RATIO=0.8
COGNATE_VAL_RATIO=0.1
COGNATE_TEST_RATIO=0.1

# Optional: External cognate lists (set to null if not using)
ADDITIONAL_TRAIN_COGNATES_SRC=null
ADDITIONAL_TRAIN_COGNATES_TGT=null
VAL_COGNATES_SRC=null
VAL_COGNATES_TGT=null
TEST_COGNATES_SRC=null
TEST_COGNATES_TGT=null
```

#### Task 2.2: Train SC Model

```bash
cd $SCIMT_DIR
bash Pipeline/train_SC_venv.sh $BASE_DIR/configs/sc/es2pt.cfg venv_sound venv_copper
```

**What happens**:
1. **Phase 1 - Cognate Extraction** (~5-30 minutes)
   - Uses FastAlign to find word pairs that are translations and orthographically similar
   - Filters pairs by Levenshtein similarity threshold (`COGNATE_THRESH`)
   - Splits cognates into train/val/test sets

2. **Phase 2 - SC Model Training** (~30 minutes - 2 hours)
   - Trains RNN model using fairseq to predict Portuguese forms from Spanish cognates
   - Learns character-level transformations (e.g., `j` → `lh`, `ñ` → `nh`)
   - Saves checkpoints based on validation loss

3. **Phase 3 - Evaluation** (~10 seconds)
   - Tests model on held-out cognate pairs
   - Computes character-level BLEU and chrF scores

**Expected output**:
```
Extracting cognates using FastAlign...
Found 3,245 cognate pairs with threshold 0.6
Split: train=2,596, val=324, test=325

Training RNN SC model...
| epoch 001 | loss 2.534 | ppl 12.60 | accuracy 0.385
| epoch 005 | loss 1.234 | ppl 3.43 | accuracy 0.612
| epoch 015 | loss 0.856 | ppl 2.35 | accuracy 0.724 (best)
done training in 1,245 seconds

Evaluating on test set...
Test BLEU (character-level): 68.4
Test chrF: 72.1
```

**Good SC model scores**:
- BLEU > 60: ✅ Excellent
- BLEU 40-60: ✅ Good
- BLEU < 40: ⚠️ Check if languages are related

**Output location**: `$BASE_DIR/models/sc_models/es_pt_RNN-0_S-1000/checkpoints/checkpoint_best.pt`

**For more details**:
- SC model parameters: [CONFIGURATION.md - SC Model Configuration](CONFIGURATION.md#sc-model-configuration)
- Monitoring training: [MONITORING.md - Monitoring SC Model Training](MONITORING.md#monitoring-sc-model-training)
- Troubleshooting: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---
**Progress**: ✅ Data Ready | ✅ Train SC Model | 🔄 Apply SC Model | ⬜ Train Tokenizer | ⬜ Train NMT | ⬜ Evaluate
---

### Phase 3: Apply SC Model to Spanish Data

```bash
bash Pipeline/pred_SC.sh $BASE_DIR/configs/sc/es2pt.cfg
```

**What happens**:
- Loads trained SC model
- Reads Spanish text from CSV files specified in `APPLY_TO`
- Transforms Spanish words to Portuguese-like forms
- Saves SC-normalized files alongside originals

**Output files** (created in `$BASE_DIR/data/raw/high-resource/`):
```
train.SC_es2pt-RNN-0-RNN-0_es2pt.src  # SC-normalized Spanish (looks like Portuguese)
train.en                               # Original English (unchanged)
```

**⚠️ IMPORTANT - SC Model ID May Change**:

The pipeline may generate files with an **extended SC model ID** (e.g., `es2pt-RNN-0-RNN-0` instead of your configured `es2pt-RNN-0`).

**After running `pred_SC.sh`**, check the actual filenames to get the full ID:
```bash
ls $BASE_DIR/data/raw/high-resource/train.SC_*
# Example output: train.SC_es2pt-RNN-0-RNN-0_es2pt.src
# Full SC model ID is: es2pt-RNN-0-RNN-0
```

Use this **full ID** in all downstream configs (tokenizer and NMT). See [TROUBLESHOOTING.md - SC Model ID Mismatch](TROUBLESHOOTING.md#sc-model-id-mismatch) for detailed diagnostic procedure if you encounter file not found errors.

### ✅ Verify Success: SC Application

Check that SC-transformed files were created:

```bash
# Check for SC-transformed training files
ls -lh $BASE_DIR/data/raw/high-resource/train.SC_*

# Should see files like:
# train.SC_es2pt-RNN-0-RNN-0_es2pt.src  (SC-transformed Spanish)
# train.en                               (original English, unchanged)
```

**Success criteria**:
- ✅ SC-transformed files exist for train, val, and test splits
- ✅ File sizes are similar to original Spanish files
- ✅ Files contain text (not empty)

**Quick validation**:
```bash
# Compare file sizes (should be similar)
wc -l $BASE_DIR/data/raw/high-resource/train.es
wc -l $BASE_DIR/data/raw/high-resource/train.SC_*.src

# Inspect a few transformed lines
head -5 $BASE_DIR/data/raw/high-resource/train.SC_*.src
```

**Troubleshooting**: If files are missing or empty, check `train_SC_venv.sh` logs for errors during SC application.

---
**Progress**: ✅ Data Ready | ✅ Train SC Model | ✅ Apply SC Model | 🔄 Train Tokenizer | ⬜ Train NMT | ⬜ Evaluate
---

### Phase 4: Train Tokenizer

#### Task 4.1: Create Tokenizer Configuration

Create `$BASE_DIR/configs/tok/es-pt_en.cfg`:

```bash
# Tokenizer Configuration
# Trains shared SentencePiece model for source (es+pt) and target (en)

# Training size
SPM_TRAIN_SIZE=1000000        # Max sentences for tokenizer training

# Languages
SRC_LANGS=es,pt               # Comma-separated source languages
SRC_TOK_NAME=es-pt            # Name for source tokenizer
TGT_LANGS=en                  # Target language
TGT_TOK_NAME=en               # Name for target tokenizer

# Data distribution (percentages must sum to 100)
# Adjust based on your data: balance low/high resource sampling
DIST=es:40,pt:10,en:50        # 40% Spanish, 10% Portuguese, 50% English

# Parallel data CSVs
TRAIN_PARALLEL=$BASE_DIR/data/csv/train.no_overlap_v1.csv
VAL_PARALLEL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
TEST_PARALLEL=$BASE_DIR/data/csv/test.csv

# Output directory
TOK_TRAIN_DATA_DIR=$BASE_DIR/models/tokenizers

# SC model ID (use full ID from Step 3)
SC_MODEL_ID=es2pt-RNN-0-RNN-0

# Tokenizer settings
VOCAB_SIZE=32000              # 32k is standard; use 16k for smaller data
SPLIT_ON_WS=false             # false = subword tokenization
INCLUDE_LANG_TOKS=true        # Add <2es>, <2pt>, <2en> tokens
INCLUDE_PAD_TOK=true
SPECIAL_TOKS=null
IS_ATT=false
```

#### Task 4.2: Train Tokenizer

```bash
bash Pipeline/train_srctgt_tokenizer.sh $BASE_DIR/configs/tok/es-pt_en.cfg
```

**What happens**:
- Samples sentences from Portuguese, Spanish, and English data according to `DIST`
- Trains SentencePiece models for source (`es-pt`) and target (`en`)
- Creates vocabulary files and tokenizer models

**Expected time**: 5-15 minutes

**Output location**: `$BASE_DIR/models/tokenizers/es-pt_en/`
```
es-pt_en.model        # SentencePiece model (pass to NMT without .model extension)
es-pt_en.vocab        # Vocabulary file
```

### ✅ Verify Success: Tokenizer Training

Check that tokenizer files were created:

```bash
# Check for tokenizer model and vocabulary
ls -lh $BASE_DIR/models/tokenizers/es-pt_en/

# Should see:
# es-pt_en.model  (~2-10 MB depending on vocab size)
# es-pt_en.vocab  (text file with vocabulary entries)
```

**Success criteria**:
- ✅ Both `.model` and `.vocab` files exist
- ✅ `.model` file size is reasonable (2-10 MB for 32k vocab)
- ✅ `.vocab` file contains expected number of entries

**Quick validation**:
```bash
# Check vocabulary size
wc -l $BASE_DIR/models/tokenizers/es-pt_en/es-pt_en.vocab
# Should show ~32000 lines (or your specified VOCAB_SIZE)

# Inspect vocabulary (should include language tokens if INCLUDE_LANG_TOKS=true)
head -20 $BASE_DIR/models/tokenizers/es-pt_en/es-pt_en.vocab
# Should see special tokens like <unk>, <s>, </s>, <2es>, <2pt>, <2en>

# Test tokenizer (requires venv_nmt)
source $SCIMT_DIR/venv_nmt/bin/activate
python -c "import sentencepiece as spm; sp = spm.SentencePieceProcessor(); sp.load('$BASE_DIR/models/tokenizers/es-pt_en/es-pt_en.model'); print(sp.encode_as_pieces('Hello world'))"
# Should output tokenized pieces
```

**Troubleshooting**: If files are missing, check tokenizer training logs for sampling or training errors.

**For more details**: [CONFIGURATION.md - Tokenizer Configuration](CONFIGURATION.md#tokenizer-configuration)

---
**Progress**: ✅ Data Ready | ✅ Train SC Model | ✅ Apply SC Model | ✅ Train Tokenizer | 🔄 Train NMT | ⬜ Evaluate
---

### Phase 5: Train NMT Model

#### Task 5.1: Create NMT Configuration

Create `$BASE_DIR/configs/nmt/pt-en.PRETRAIN.yaml`:

```yaml
# NMT Configuration: Portuguese → English with SC Augmentation
# ⚠️ CRITICAL: YAML does NOT expand environment variables
# Replace ALL paths below with absolute paths (not $BASE_DIR)

# Output and evaluation
src: pt
tgt: en
save: /absolute/path/to/models/nmt_models/pt-en/PRETRAIN   # ⚠️ Use absolute path
test_checkpoint: null
remove_special_toks: true
verbose: false
little_verbose: true

# Fine-tuning (set to null for training from scratch)
from_pretrained: null

# Data (absolute paths required)
train_data: /absolute/path/to/data/csv/train.no_overlap_v1.csv
val_data: /absolute/path/to/data/csv/val.no_overlap_v1.csv
test_data: /absolute/path/to/data/csv/test.csv
append_src_token: false
append_tgt_token: false
upsample: false
sc_model_id: es2pt-RNN-0-RNN-0   # Use full SC model ID from Step 3

# Tokenizer (absolute path without .model extension)
spm: /absolute/path/to/models/tokenizers/es-pt_en/es-pt_en
do_char: false

# Training hyperparameters
n_gpus: 1
seed: 1000
max_steps: 50000              # 50k for quick test; 250k for full training
train_batch_size: 32          # Reduce to 16 if GPU memory issues
val_batch_size: 32
test_batch_size: 32
early_stop: 10                # Stop if no improvement for 10 epochs
save_top_k: 5                 # Keep 5 best checkpoints
val_interval: 0.5             # Validate twice per epoch
learning_rate: 2e-04
weight_decay: 0.01
device: cuda                  # Use 'cpu' if no GPU

# Model architecture (base model)
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

**⚠️ Replace all `/absolute/path/to/` with your actual paths**. You can use `$BASE_DIR` in your shell to construct paths:

```bash
echo "save: $BASE_DIR/models/nmt_models/pt-en/PRETRAIN"
echo "train_data: $BASE_DIR/data/csv/train.no_overlap_v1.csv"
echo "spm: $BASE_DIR/models/tokenizers/es-pt_en/es-pt_en"
```

Then copy the expanded paths into the YAML file.

#### Task 5.2: Train NMT Model

```bash
cd $SCIMT_DIR/NMT
python train.py -c $BASE_DIR/configs/nmt/pt-en.PRETRAIN.yaml -m TRAIN
```

**What happens**:
- Loads tokenizer and training data (Portuguese-English + SC-normalized Spanish-English)
- Initializes Transformer model
- Trains with validation checks at intervals specified by `val_interval`
- Saves checkpoints based on validation loss
- Applies early stopping when validation doesn't improve

**Expected time**:
- 50k steps, batch size 32, 1 GPU: ~6-12 hours
- 250k steps, batch size 32, 1 GPU: ~24-48 hours
- Multi-GPU training is faster

**Monitoring options**:

**Option 1: Real-time CSV metrics**
```bash
# In another terminal
tail -f $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/logs/version_0/metrics.csv
```

**Option 2: TensorBoard**
```bash
# In another terminal
source $SCIMT_DIR/venv_sound/bin/activate
tensorboard --logdir $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/logs
# Open browser to http://localhost:6006/
```

**Output location**: `$BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/checkpoints/`

**For more details**:
- NMT parameters: [CONFIGURATION.md - NMT Configuration](CONFIGURATION.md#nmt-configuration)
- Monitoring training: [MONITORING.md - Monitoring NMT Training](MONITORING.md#monitoring-nmt-training)

### ✅ Verify Success: NMT Training

Check that training completed successfully:

```bash
# Check for saved checkpoints
ls -lh $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/checkpoints/

# Should see multiple checkpoint files:
# epoch=X-step=Y.ckpt (multiple checkpoints)
# best.ckpt or best_model.ckpt (best checkpoint based on validation loss)
```

**Success criteria**:
- ✅ At least one checkpoint file exists
- ✅ Training logs show decreasing loss over time
- ✅ No error messages in terminal output
- ✅ Validation BLEU scores are reasonable (> 5 for low-resource)

**Quick validation**:
```bash
# Check training logs for final metrics
tail -50 $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/logs/version_0/metrics.csv

# Look for:
# - Decreasing train_loss
# - Stable or improving val_loss
# - val_bleu improving over epochs

# Check for early stopping or completion
grep -i "early\|finish\|complete" $SCIMT_DIR/NMT/*.log
```

**Troubleshooting**:
- **No checkpoints**: Training may have failed - check terminal output for CUDA/memory errors
- **High loss (>5.0 after 10k steps)**: Check tokenizer paths, data loading, or reduce learning rate
- **See [MONITORING.md](MONITORING.md#monitoring-nmt-training)** for detailed troubleshooting

---
**Progress**: ✅ Data Ready | ✅ Train SC Model | ✅ Apply SC Model | ✅ Train Tokenizer | ✅ Train NMT | 🔄 Evaluate
---

### Phase 6: Evaluate

#### Task 6.1: Evaluate NMT Model

```bash
python train.py -c $BASE_DIR/configs/nmt/pt-en.PRETRAIN.yaml -m TEST
```

**What happens**:
- Loads best checkpoint from training
- Runs inference on test set
- Computes BLEU and chrF scores
- Saves predictions to predictions directory

**Expected output**:
```
Loading checkpoint: .../checkpoints/best_model.ckpt
Running inference on test set...
Test BLEU: 28.6
Test chrF: 56.3
Predictions saved to: .../predictions/
```

**Output files** (`$BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/predictions/`):
```
test_predictions.txt    # Model translations
test_references.txt     # Ground truth translations
test_sources.txt        # Source sentences
metrics.json            # BLEU, chrF, loss
```

**For detailed score ranges by data size**, see [MONITORING.md - Typical Score Ranges](MONITORING.md#typical-score-ranges).

**For detailed evaluation guidance**, see [MONITORING.md - Evaluating NMT Models](MONITORING.md#evaluating-nmt-models).

#### Task 6.2: Compare with Baseline (Optional)

To measure SC augmentation impact, train a baseline model without SC:

1. Create config: `pt-en.BASELINE.yaml` (same as PRETRAIN but with `sc_model_id: null`)
2. Train: `python train.py -c pt-en.BASELINE.yaml -m TRAIN`
3. Evaluate: `python train.py -c pt-en.BASELINE.yaml -m TEST`
4. Compare BLEU scores:

```
Baseline (no SC):     BLEU 18.3
SC-Augmented:         BLEU 28.6
Improvement:          +10.3 BLEU points (56% relative improvement)
```

---

## Next Steps

### Experiment with Configuration

Now that you've run the complete workflow, try adjusting parameters:

**SC Model Tuning**:
- **Cognate threshold**: Lower = stricter (0.5), Higher = more pairs (0.7)
- **Model type**: Try `SC_MODEL_TYPE=SMT` (faster, sometimes better for distant languages)
- **Hyperparameters**: Experiment with different `RNN_HYPERPARAMS_ID` values

**Tokenizer Tuning**:
- **Vocabulary size**: 16k (smaller data), 32k (standard), 64k (large data)
- **Distribution**: Adjust `DIST` to balance low/high resource sampling

**NMT Tuning**:
- **Training steps**: 50k (quick), 100k (moderate), 250k (full)
- **Model size**: Increase `d_model`, `encoder_layers`, `decoder_layers` for larger models
- **Batch size**: Increase for faster training (if GPU memory allows)

**See [CONFIGURATION.md](CONFIGURATION.md)** for complete parameter reference.

### Analyze Model Quality

**Quantitative analysis**:
- Compare BLEU/chrF scores across configurations
- Test on domain-specific test sets
- Measure improvement over baseline

**Qualitative analysis**:
- Inspect predictions manually
- Identify common error types (hallucination, omission, mistranslation)
- Check fluency and adequacy

**See [MONITORING.md - Evaluating NMT Models](MONITORING.md#evaluating-nmt-models)** for detailed evaluation techniques.

### Scale to New Language Pairs

To apply CharLOTTE to your own languages:

1. **Identify language triplet**:
   - Low-resource pair: X → English
   - High-resource related pair: Y → English
   - Ensure X and Y are linguistically related

2. **Gather data**:
   - See [DATA_PREPARATION.md](DATA_PREPARATION.md) for obtaining corpora
   - Minimum: 5k X-en pairs, 100k Y-en pairs

3. **Follow this workflow**:
   - Train Y→X SC model
   - Apply to Y-en data
   - Train X-en NMT with augmented data

### Troubleshooting

If you encounter issues:

1. **Installation problems**: [SETUP.md](SETUP.md)
2. **Training errors**: [TROUBLESHOOTING.md - Training Issues](TROUBLESHOOTING.md#training-issues)
3. **Evaluation issues**: [TROUBLESHOOTING.md - Evaluation Issues](TROUBLESHOOTING.md#evaluation-issues)
4. **Poor results**: [MONITORING.md - Troubleshooting](MONITORING.md#troubleshooting)

---

## Summary

You've completed the full CharLOTTE workflow:

✅ **Prepared data** in CSV format
✅ **Trained SC model** to learn character correspondences
✅ **Applied SC model** to augment training data
✅ **Trained tokenizer** on combined data
✅ **Trained NMT model** with SC augmentation
✅ **Evaluated** translation quality

**Expected gains from SC augmentation**: 5-15 BLEU points for related language pairs in low-resource settings (5k-30k pairs).

**Key takeaways**:
- SC augmentation works best for related languages with systematic correspondences
- Data quality matters more than quantity
- Monitor training carefully to detect issues early
- Always compare with a baseline to measure SC impact

---

**[← Back to README](../README.md)** | **[Data Preparation →](DATA_PREPARATION.md)** | **[Configuration →](CONFIGURATION.md)** | **[Monitoring →](MONITORING.md)** | **[Troubleshooting →](TROUBLESHOOTING.md)**
