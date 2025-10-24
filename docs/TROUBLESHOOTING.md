# CharLOTTE Troubleshooting Guide

**Purpose:** Common issues and solutions for CharLOTTE installation, training, and evaluation.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Common Failure Modes](#common-failure-modes)
3. [Runtime Issues](#runtime-issues)
   - [Training Issues](#training-issues)
   - [Evaluation Issues](#evaluation-issues)
4. [Getting Help](#getting-help)

---

## Installation Issues

### Python Version Issues

If you see errors about package versions not being available, ensure you're using Python 3.10 or higher:
```bash
python --version  # Should show 3.10.0 or higher
```

If your system Python is too old, install Python 3.10+ using:
- **macOS**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt install python3.11 python3.11-venv`
- **Windows**: Download from https://www.python.org/downloads/

### CUDA/GPU Issues

If PyTorch doesn't detect your GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If this returns `False` and you have a CUDA GPU, reinstall PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Package Conflicts

If you encounter dependency conflicts:
1. Create a fresh virtual environment
2. Install `requirements-minimal.txt` first
3. Add additional packages as needed

### fairseq Installation Failures

If fairseq installation fails with permission errors (common on macOS):

**Error**: `PermissionError: [Errno 1] Operation not permitted: 'fairseq/examples'`

**Solution**: Install from git instead of PyPI:
```bash
# In copper environment
pip install cython "numpy<1.24,>=1.21" torch==2.4.1
pip install git+https://github.com/pytorch/fairseq.git@v0.10.2
```

**Verify installation**:
```bash
python -c "import fairseq; print(fairseq.__version__)"  # Should print: 0.10.2
```

### FastAlign Build Issues

If FastAlign fails to build with CMake 4.x:

**Error**: `CMake Error: Compatibility with CMake < 3.5 has been removed`

**Solution**: Edit `CMakeLists.txt` in the fast_align directory:
```cmake
# Change line 2 from:
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# To:
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

Then build:
```bash
cd fast_align
cmake . && make
cp fast_align ~/bin/  # or your preferred location
cp atools ~/bin/
```

### Missing NLP Dependencies for SC Training

If you see `ModuleNotFoundError` for nltk, spacy, indicnlp, or Levenshtein when running SC training:

**Solution**: These are required for cognate extraction but may not be in older requirements files:
```bash
# In sound environment
pip install nltk spacy indic-nlp-library python-Levenshtein
python -m spacy download es_core_news_sm
python -m spacy download xx_sent_ud_sm
```

These dependencies are included in the updated `requirements-minimal.txt`.

### File Path Issues in Configuration Files

**Problem**: Scripts fail with `FileNotFoundError` even though files exist

**Common causes**: Using `~` (tilde) or relative paths instead of absolute paths.

**Solution**: See [SETUP.md - Path Configuration Guide](SETUP.md#path-configuration-guide) for complete guidance on:
- When variables expand (shell commands, .cfg files)
- When you need absolute paths (YAML files)
- How to set up environment variables

### Moses Decoder Not Installed

**Problem**: SC training fails with errors like:
```
/path/to/CopperMT/submodules/mosesdecoder/scripts/training/clean-corpus-n.perl: No such file or directory
/path/to/CopperMT/submodules/mosesdecoder/bin/build_binary: No such file or directory
```

**Cause**: Moses decoder submodules not initialized or built

**Solution**: Complete the CopperMT installation including Moses submodules (see [SETUP.md](SETUP.md#installing-coppermt)):
```bash
cd CopperMT/CopperMT
git submodule init
git submodule update
cd submodules/mgiza/mgizapp && cmake . && make && make install && cp scripts/merge_alignment.py bin/
cd ../../mosesdecoder && bjam -j4 -q -d2
```

**Note**: You must install Boost (≥1.64) before building Moses.

### FastAlign Location Issues

**Problem**: `train_SC_venv.sh` fails with "fast_align: No such file or directory"

**Cause**: Script expects FastAlign at `./../fast_align/build/` relative to SCIMT root

**Solutions**:
1. **Install at expected location**:
   ```bash
   mkdir -p /path/to/SCIMT/../fast_align/build
   cp ~/bin/fast_align /path/to/SCIMT/../fast_align/build/
   cp ~/bin/atools /path/to/SCIMT/../fast_align/build/
   ```

2. **Use symlinks**:
   ```bash
   mkdir -p /path/to/SCIMT/../fast_align/build
   ln -s ~/bin/fast_align /path/to/SCIMT/../fast_align/build/fast_align
   ln -s ~/bin/atools /path/to/SCIMT/../fast_align/build/atools
   ```

---

## Runtime Issues

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

**Solution**: YAML files don't expand variables! See [SETUP.md - Path Configuration Guide](SETUP.md#path-configuration-guide) for the complete explanation of which file types expand variables and which require absolute paths.

### SC Model ID Mismatch

**Error**: NMT training or tokenizer training can't find SC-normalized data files

**Common symptoms**:
```
FileNotFoundError: No such file or directory: '.../train.SC_es2pt-RNN-0_es2pt.src'
KeyError: 'SC_es2pt-RNN-0' not found in data files
```

**Root cause**: The SC pipeline generates files with an **extended model ID** that differs from what you specified in your SC config.

**Example scenario**:
1. You set `SC_MODEL_ID=es2pt-RNN-0` in your SC config
2. After running `pred_SC.sh`, the actual files are named: `train.SC_es2pt-RNN-0-RNN-0_es2pt.src`
3. The **full SC model ID** is `es2pt-RNN-0-RNN-0` (with extra `-RNN-0` appended)

**Solution - Find the actual SC model ID**:

Run this diagnostic command to discover the actual SC model ID from your generated files:

```bash
# Navigate to your high-resource data directory
cd $BASE_DIR/data/raw/high-resource/

# List SC-transformed files
ls -1 train.SC_*

# Example output:
# train.SC_es2pt-RNN-0-RNN-0_es2pt.src
#         ^^^^^^^^^^^^^^^^^^^^^
#         This is your full SC model ID

# Extract the SC model ID automatically
ls train.SC_* | head -1 | sed 's/.*SC_\(.*\)_[^_]*\.src/\1/'
# Output: es2pt-RNN-0-RNN-0
```

**Then update all downstream configs**:

1. **Tokenizer config** (`configs/tok/your-tokenizer.cfg`):
   ```bash
   SC_MODEL_ID=es2pt-RNN-0-RNN-0  # Use the FULL ID from filenames
   ```

2. **NMT config** (`configs/nmt/your-nmt.yaml`):
   ```yaml
   sc_model_id: es2pt-RNN-0-RNN-0  # Must match tokenizer config
   ```

**Verification**:

After updating configs, verify the SC model ID is consistent:

```bash
# Check what files were actually created
ls $BASE_DIR/data/raw/high-resource/train.SC_*

# Verify tokenizer config
grep "SC_MODEL_ID" $BASE_DIR/configs/tok/your-tokenizer.cfg

# Verify NMT config
grep "sc_model_id" $BASE_DIR/configs/nmt/your-nmt.yaml
```

All three should show the **same full SC model ID** (e.g., `es2pt-RNN-0-RNN-0`).

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

---

## Common Failure Modes

### 1. fast_align: command not found
**What it looks like:**
```
Error: fast_align: command not found
```

**Fix**: See [Installing FastAlign in SETUP.md](SETUP.md#installing-fastalign)

### 2. Moses decoder scripts missing
**What it looks like:**
```
/path/to/CopperMT/submodules/mosesdecoder/scripts/training/clean-corpus-n.perl: No such file or directory
```

**Fix**: Complete CopperMT installation in [SETUP.md](SETUP.md#installing-coppermt)

### 3. ImportError: cannot import name 'TOKENIZERS'
**What it looks like:**
```
ImportError: cannot import name 'TOKENIZERS' from 'sacrebleu.tokenizers'
```

**Fix**: Wrong sacrebleu version
```bash
source venv_copper/bin/activate
pip install 'sacrebleu==1.5.1'
```

### 4. TypeError: 'Optional[str]'
**What it looks like:**
```
TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
```

**Fix**: Python 3.10 incompatibility - fairseq patches should have been applied during setup

---

## Runtime Issues

### Training Issues

#### Problem: Loss Not Decreasing

**Symptoms:**
```
Epoch 0: train_loss=3.456, val_loss=3.234
Epoch 1: train_loss=3.421, val_loss=3.198
Epoch 2: train_loss=3.389, val_loss=3.156
...
Epoch 10: train_loss=3.012, val_loss=2.987  # Still high
```

**Solutions:**
- Decrease learning rate (try `1e-04` instead of `2e-04`)
- Check data quality (ensure CSV paths are correct)
- Verify tokenizer is working (check tokenized samples)
- Increase model capacity (more layers or larger `d_model`)

#### Problem: Validation Loss Diverging

**Symptoms:**
```
Epoch 5: train_loss=2.123, val_loss=2.456
Epoch 6: train_loss=2.012, val_loss=2.543
Epoch 7: train_loss=1.932, val_loss=2.678  # Diverging
```

**Solutions:**
- **Overfitting**: Add dropout (`dropout=0.2` or higher)
- Reduce model size (fewer layers or smaller `d_model`)
- Early stopping will trigger automatically (default patience=10 epochs)
- Consider data augmentation or more training data

#### Problem: CUDA Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB (GPU 0; 11.91 GiB total capacity; 9.87 GiB already allocated)
```

**Solutions:**
```yaml
# Reduce batch size in config
train_batch_size: 16  # Down from 32
val_batch_size: 16

# Or reduce model size
encoder_layers: 4  # Down from 6
decoder_layers: 4
d_model: 256      # Down from 512
```

#### Problem: Training Too Slow

**Expected speeds:**
- **GPU training**: 100-500 steps/minute (depends on batch size and GPU)
- **CPU training**: 1-10 steps/minute (much slower)

**Optimizations:**
- Increase `train_batch_size` (if GPU memory allows)
- Use multiple GPUs (`n_gpus: 2` or more)
- Reduce `val_interval` (validate less frequently, e.g., `1.0` instead of `0.5`)
- Use fp16 training (requires code modification)

### Evaluation Issues

#### BLEU < 5 (Model Not Learning)

**Causes and Solutions:**
- Check data loading: Verify CSV paths are correct
- Check tokenization: Ensure tokenizer is applied correctly
- Check training: Model may have crashed early (check logs)
- Check evaluation: Test set may not match training domain

#### BLEU 5-15 (Model Learning Slowly)

**Causes and Solutions:**
- Increase training steps: Try 100k or 250k steps
- Increase data: SC augmentation may not provide enough signal
- Check architecture: May need larger model (more layers or d_model)
- Check SC model quality: Low SC BLEU means poor augmentation

#### BLEU plateaued at 20-25 (Data Limitation)

**Causes and Solutions:**
- Data size may be limiting factor
- Consider:
  - Using larger high-resource dataset for SC augmentation
  - Collecting more low-resource parallel data
  - Pre-training on related language pairs
  - Using multilingual models

#### BLEU worse with SC augmentation

**Causes and Solutions:**
- SC model may be introducing noise
- Check:
  - SC model test BLEU (should be > 30)
  - Language pair relatedness (SC only helps related languages)
  - SC_MODEL_ID matching between configs
  - Data distribution in tokenizer config (balance high/low resource)

---

## Getting Help

If you've tried the solutions above and still have issues:

1. Check the [Setup Guide](SETUP.md) - you may have skipped a step
2. Run the Pre-Flight Check to identify missing components
3. Search existing GitHub issues
4. Open a new issue with:
   - Your Python version (`python --version`)
   - Your OS and version
   - Complete error message and stack trace
   - Steps to reproduce

---

**[← Back to README](../README.md)** | **[← Setup Guide](SETUP.md)** | **[← Experimentation Guide](EXPERIMENTATION.md)**
