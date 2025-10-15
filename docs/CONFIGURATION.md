# CharLOTTE Configuration Reference

**Purpose:** Complete parameter reference for SC model, tokenizer, and NMT configuration files.

**Prerequisites**: Review [EXPERIMENTATION.md](EXPERIMENTATION.md) for context on when and how these configurations are used.

---

## Table of Contents

1. [Configuration File Types](#configuration-file-types)
2. [Path Configuration Guidelines](#path-configuration-guidelines)
3. [SC Model Configuration (.cfg)](#sc-model-configuration-cfg)
4. [Tokenizer Configuration (.cfg)](#tokenizer-configuration-cfg)
5. [NMT Configuration (.yaml)](#nmt-configuration-yaml)
6. [Configuration Templates](#configuration-templates)

---

## Configuration File Types

CharLOTTE uses two configuration file formats:

| File Type | Used For | Variable Expansion | Path Requirements |
|-----------|----------|-------------------|-------------------|
| **Shell Config (.cfg)** | SC models, tokenizers | ✅ YES | `$VARIABLE` or absolute |
| **YAML Config (.yaml)** | NMT training | ❌ NO | **Absolute only** |

**Important**: For complete path configuration guidance, see [SETUP.md - Path Configuration Guide](SETUP.md#path-configuration-guide)

---

## SC Model Configuration (.cfg)

Sound Correspondence model configuration for cognate prediction.

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `MODULE_HOME_DIR` | Path to SCIMT root directory | `/home/user/SCIMT` or `$SCIMT_DIR` |
| `SRC` | Source language code (high-resource) | `es` (Spanish) |
| `TGT` | Target language code (low-resource) | `pt` (Portuguese) |
| `SEED` | Random seed for reproducibility | `1000` |
| `PARALLEL_TRAIN` | Path to training data CSV | `$BASE_DIR/data/csv/train.no_overlap_v1.csv` |
| `PARALLEL_VAL` | Path to validation data CSV | `$BASE_DIR/data/csv/val.no_overlap_v1.csv` |
| `PARALLEL_TEST` | Path to test data CSV | `$BASE_DIR/data/csv/test.csv` |
| `SC_MODEL_TYPE` | Model architecture | `RNN` or `SMT` |
| `COPPERMT_DIR` | Path to CopperMT installation | `$SCIMT_DIR/CopperMT/CopperMT` |
| `COPPERMT_DATA_DIR` | Output directory for SC models | `$BASE_DIR/models/sc_models` |
| `COGNATE_TRAIN` | Output directory for cognate lists | `$BASE_DIR/models/sc_models/cognates` |
| `SC_MODEL_ID` | Unique identifier for this SC model | `es2pt-RNN-0` |

### Cognate Extraction Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `COGNATE_THRESH` | Levenshtein similarity threshold (0.0-1.0) | `0.5` (strict) to `0.7` (lenient) |
| `COGNATE_TRAIN_RATIO` | Proportion for training | `0.8` |
| `COGNATE_VAL_RATIO` | Proportion for validation | `0.1` |
| `COGNATE_TEST_RATIO` | Proportion for test | `0.1` |
| `REVERSE_SRC_TGT_COGNATES` | Swap source/target for cognates | `false` |

**Choosing `COGNATE_THRESH`**:
- `0.5`: Strict - only very similar cognates (closely-related languages)
- `0.6`: Balanced - **recommended starting point**
- `0.7`: Lenient - more pairs (distantly-related languages)

### RNN-Specific Parameters

Only used when `SC_MODEL_TYPE=RNN`:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `RNN_HYPERPARAMS` | Path to hyperparameter directory | `$MODULE_HOME_DIR/Pipeline/parameters/rnn_hyperparams` |
| `RNN_HYPERPARAMS_ID` | Hyperparameter set ID | `0` (default), `1`, `2`, etc. |
| `BEAM` | Beam search width | `5` |
| `NBEST` | N-best list size | `1` |

**Available Hyperparameter IDs**:
```bash
ls $SCIMT_DIR/Pipeline/parameters/rnn_hyperparams/
# Shows: manifest.json, 0, 1, 2, ... (each ID is a different configuration)
```

- `0`: Default hyperparameters (good starting point)
- Check `manifest.json` for details on each ID

### Output Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `APPLY_TO` | Comma-separated CSVs to normalize | `$PARALLEL_TRAIN,$PARALLEL_VAL,$PARALLEL_TEST` |
| `PARAMETERS_DIR` | Directory for CopperMT parameters | `$BASE_DIR/configs/sc/parameters` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NO_GROUPING` | Disable grouping (recommended: true) | `true` |
| `ADDITIONAL_TRAIN_COGNATES_SRC` | Extra cognate list (source) | `null` |
| `ADDITIONAL_TRAIN_COGNATES_TGT` | Extra cognate list (target) | `null` |
| `VAL_COGNATES_SRC` | Custom validation cognates | `null` |
| `VAL_COGNATES_TGT` | Custom validation cognates | `null` |
| `TEST_COGNATES_SRC` | Custom test cognates | `null` |
| `TEST_COGNATES_TGT` | Custom test cognates | `null` |

### Complete SC Config Example

```bash
# SC Model Configuration: Spanish → Portuguese

# Module paths
MODULE_HOME_DIR=$SCIMT_DIR

# SC model language pair
SRC=es
TGT=pt
SEED=1000

# Data paths
PARALLEL_TRAIN=$BASE_DIR/data/csv/train.no_overlap_v1.csv
PARALLEL_VAL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
PARALLEL_TEST=$BASE_DIR/data/csv/test.csv
APPLY_TO=$PARALLEL_TRAIN,$PARALLEL_VAL,$PARALLEL_TEST

# SC model settings
NO_GROUPING=true
SC_MODEL_TYPE=RNN
COGNATE_THRESH=0.6

# Output directories
COGNATE_TRAIN=$BASE_DIR/models/sc_models/cognates
COPPERMT_DATA_DIR=$BASE_DIR/models/sc_models
COPPERMT_DIR=$SCIMT_DIR/CopperMT/CopperMT
PARAMETERS_DIR=$BASE_DIR/configs/sc/parameters

# RNN hyperparameters
RNN_HYPERPARAMS=$MODULE_HOME_DIR/Pipeline/parameters/rnn_hyperparams
RNN_HYPERPARAMS_ID=0
BEAM=5
NBEST=1

# Cognate extraction settings
REVERSE_SRC_TGT_COGNATES=false
SC_MODEL_ID=es2pt-RNN-0
COGNATE_TRAIN_RATIO=0.8
COGNATE_VAL_RATIO=0.1
COGNATE_TEST_RATIO=0.1

# Additional cognate data (optional)
ADDITIONAL_TRAIN_COGNATES_SRC=null
ADDITIONAL_TRAIN_COGNATES_TGT=null
VAL_COGNATES_SRC=null
VAL_COGNATES_TGT=null
TEST_COGNATES_SRC=null
TEST_COGNATES_TGT=null
```

---

## Tokenizer Configuration (.cfg)

SentencePiece tokenizer training configuration.

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `SRC_LANGS` | Comma-separated source language codes | `es,pt` |
| `SRC_TOK_NAME` | Name for source tokenizer | `es-pt` |
| `TGT_LANGS` | Comma-separated target language codes | `en` |
| `TGT_TOK_NAME` | Name for target tokenizer | `en` |
| `TRAIN_PARALLEL` | Training data CSV | `$BASE_DIR/data/csv/train.no_overlap_v1.csv` |
| `VAL_PARALLEL` | Validation data CSV | `$BASE_DIR/data/csv/val.no_overlap_v1.csv` |
| `TEST_PARALLEL` | Test data CSV | `$BASE_DIR/data/csv/test.csv` |
| `TOK_TRAIN_DATA_DIR` | Output directory | `$BASE_DIR/models/tokenizers` |
| `VOCAB_SIZE` | Vocabulary size | `32000` (typical) |

### Data Distribution Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `DIST` | Language distribution (percentages must sum to 100) | `es:40,pt:10,en:50` |
| `SPM_TRAIN_SIZE` | Max sentences for tokenizer training | `1000000` |

**Choosing `DIST` values**:
- Balance based on data availability
- Example: If you have 200k Spanish, 10k Portuguese, 210k English:
  - Calculate proportions: es=200k/(420k)≈48%, pt=10k/(420k)≈2%, en=210k/(420k)≈50%
  - Round and adjust: `es:48,pt:2,en:50` or `es:40,pt:10,en:50` (boost low-resource)

### Special Token Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `INCLUDE_LANG_TOKS` | Add language tokens (`<2es>`, `<2pt>`) | `true` |
| `INCLUDE_PAD_TOK` | Add padding token | `true` |
| `SPLIT_ON_WS` | Force splits on whitespace | `false` |
| `SPECIAL_TOKS` | Custom special tokens (comma-separated) | `null` |
| `IS_ATT` | Add attention-specific tokens | `false` |

### SC Integration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `SC_MODEL_ID` | Use SC-normalized data | `es2pt-RNN-0-RNN-0` or `null` |

**Important**: When using SC-normalized data, the `SC_MODEL_ID` must match the **full ID** from the SC-normalized filenames:
- After `pred_SC.sh`, check filenames: `train.SC_es2pt-RNN-0-RNN-0_es2pt.src`
- Use full ID: `SC_MODEL_ID=es2pt-RNN-0-RNN-0` (includes suffix)

### Complete Tokenizer Config Example

```bash
# Tokenizer Configuration

# Training size
SPM_TRAIN_SIZE=1000000

# Languages
SRC_LANGS=es,pt
SRC_TOK_NAME=es-pt
TGT_LANGS=en
TGT_TOK_NAME=en

# Data distribution
DIST=es:40,pt:10,en:50

# Parallel data CSVs
TRAIN_PARALLEL=$BASE_DIR/data/csv/train.no_overlap_v1.csv
VAL_PARALLEL=$BASE_DIR/data/csv/val.no_overlap_v1.csv
TEST_PARALLEL=$BASE_DIR/data/csv/test.csv

# Output directory
TOK_TRAIN_DATA_DIR=$BASE_DIR/models/tokenizers

# SC model ID (use full ID from normalized filenames)
SC_MODEL_ID=es2pt-RNN-0-RNN-0

# Tokenizer settings
VOCAB_SIZE=32000
SPLIT_ON_WS=false
INCLUDE_LANG_TOKS=true
INCLUDE_PAD_TOK=true
SPECIAL_TOKS=null
IS_ATT=false
```

---

## NMT Configuration (.yaml)

Neural Machine Translation training configuration.

### Output and Evaluation Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `src` | Source language code | `pt` |
| `tgt` | Target language code | `en` |
| `save` | Output directory (**must be absolute**) | `/home/user/project/models/nmt_models/pt-en/PRETRAIN` |
| `test_checkpoint` | Checkpoint for testing (`null` = latest) | `null` |
| `remove_special_toks` | Remove special tokens from outputs | `true` |
| `verbose` | Detailed logging | `false` |
| `little_verbose` | Moderate logging | `true` |

### Data Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `train_data` | Training CSV (**absolute path**) | `/absolute/path/data/csv/train.no_overlap_v1.csv` |
| `val_data` | Validation CSV (**absolute path**) | `/absolute/path/data/csv/val.no_overlap_v1.csv` |
| `test_data` | Test CSV (**absolute path**) | `/absolute/path/data/csv/test.csv` |
| `append_src_token` | Add source language token to input | `false` |
| `append_tgt_token` | Add target language token to input | `false` |
| `upsample` | Upsample minority languages | `false` |
| `sc_model_id` | Use SC-normalized data (`null` = none) | `es2pt-RNN-0-RNN-0` or `null` |

### Tokenizer Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `spm` | Path to SentencePiece model (**without .model extension**) | `/absolute/path/tokenizers/es-pt_en/es-pt_en` |
| `do_char` | Use character-level tokenization | `false` |

### Training Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `n_gpus` | Number of GPUs | `1` (single), `2-8` (multi-GPU) |
| `seed` | Random seed | `1000`, `42`, `123` |
| `max_steps` | Maximum training steps | `50000` (quick), `250000` (full) |
| `train_batch_size` | Training batch size | `16-128` (GPU memory dependent) |
| `val_batch_size` | Validation batch size | Same as train |
| `test_batch_size` | Test batch size | Same as train |
| `early_stop` | Early stopping patience (epochs) | `10` |
| `save_top_k` | Number of checkpoints to keep | `5` |
| `val_interval` | Validation frequency (fraction of epoch) | `0.5` (twice), `1.0` (once) |
| `learning_rate` | Learning rate | `2e-04` (typical), `1e-04` to `5e-04` |
| `weight_decay` | L2 regularization | `0.01` |
| `device` | Device type | `cuda` or `cpu` |

### Model Architecture Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `encoder_layers` | Encoder depth | `6` (base), `12` (large) |
| `encoder_attention_heads` | Encoder attention heads | `8` (base), `16` (large) |
| `encoder_ffn_dim` | Encoder feed-forward dimension | `2048` (base), `4096` (large) |
| `encoder_layerdrop` | Encoder layer dropout | `0.0` (none), `0.1-0.2` (regularization) |
| `decoder_layers` | Decoder depth | `6` (base), `12` (large) |
| `decoder_attention_heads` | Decoder attention heads | `8` (base), `16` (large) |
| `decoder_ffn_dim` | Decoder feed-forward dimension | `2048` (base), `4096` (large) |
| `decoder_layerdrop` | Decoder layer dropout | `0.0` (none), `0.1-0.2` (regularization) |
| `max_position_embeddings` | Maximum sequence length | `512` (typical), `1024` (long) |
| `max_length` | Maximum generation length | Same as max_position_embeddings |
| `d_model` | Model dimension | `512` (base), `768` or `1024` (large) |
| `dropout` | General dropout rate | `0.1` (typical), `0.2-0.3` (more regularization) |
| `activation_function` | Activation function | `gelu`, `relu` |

### Fine-tuning Parameter

| Parameter | Description | Example |
|-----------|-------------|---------|
| `from_pretrained` | Path to pretrained checkpoint | `null` (train from scratch) or `/path/to/checkpoint.ckpt` |

### Complete NMT Config Example

```yaml
# NMT Configuration: Portuguese → English

# ⚠️ CRITICAL: YAML does NOT expand variables
# Replace ALL $VARIABLE with absolute paths

# outputs
src: pt
tgt: en
save: /home/username/charlotte-project/models/nmt_models/pt-en/PRETRAIN
test_checkpoint: null
remove_special_toks: true
verbose: false
little_verbose: true

# finetune?
from_pretrained: null

# data (MUST be absolute paths)
train_data: /home/username/charlotte-project/data/csv/train.no_overlap_v1.csv
val_data: /home/username/charlotte-project/data/csv/val.no_overlap_v1.csv
test_data: /home/username/charlotte-project/data/csv/test.csv
append_src_token: false
append_tgt_token: false
upsample: false
sc_model_id: es2pt-RNN-0-RNN-0

# tokenizers (MUST be absolute path, without .model extension)
spm: /home/username/charlotte-project/models/tokenizers/es-pt_en/es-pt_en
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

# model architecture (base model)
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

---

## Configuration Templates

### Template 1: Low-Resource (5k sentences)

**SC Config**:
```bash
SRC=es
TGT=pt
SC_MODEL_TYPE=RNN
COGNATE_THRESH=0.6  # Balanced
RNN_HYPERPARAMS_ID=0  # Default
```

**NMT Config**:
```yaml
max_steps: 50000  # Shorter training
train_batch_size: 16  # Smaller batch
encoder_layers: 4  # Smaller model
decoder_layers: 4
d_model: 256
```

### Template 2: Medium-Resource (50k sentences)

**SC Config**:
```bash
SRC=es
TGT=pt
SC_MODEL_TYPE=RNN
COGNATE_THRESH=0.5  # Stricter
RNN_HYPERPARAMS_ID=0
```

**NMT Config**:
```yaml
max_steps: 100000  # Standard training
train_batch_size: 32  # Standard batch
encoder_layers: 6  # Base model
decoder_layers: 6
d_model: 512
```

### Template 3: High-Quality (100k+ sentences)

**SC Config**:
```bash
SRC=es
TGT=pt
SC_MODEL_TYPE=RNN
COGNATE_THRESH=0.5
RNN_HYPERPARAMS_ID=0
```

**NMT Config**:
```yaml
max_steps: 250000  # Extended training
train_batch_size: 64  # Larger batch (if GPU allows)
encoder_layers: 12  # Large model
decoder_layers: 12
encoder_ffn_dim: 4096
decoder_ffn_dim: 4096
d_model: 768
```

---

## Next Steps

- **Run experiments** → [EXPERIMENTATION.md](EXPERIMENTATION.md)
- **Monitor training** → [MONITORING.md](MONITORING.md)
- **Troubleshoot issues** → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**[← Back to README](../README.md)** | **[Experiments →](EXPERIMENTATION.md)** | **[Monitoring →](MONITORING.md)**
