# CharLOTTE Setup Guide

**Purpose:** Complete installation and configuration instructions for CharLOTTE.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing FastAlign](#installing-fastalign)
3. [Path Configuration Guide](#path-configuration-guide) ⭐ NEW
4. [Installation](#installation)
5. [Installing CopperMT](#installing-coppermt)
6. [Pre-Flight Check](#pre-flight-check) ⭐ NEW
7. [Quick Test](#quick-test)
8. [Installation Verification](#installation-verification) ⭐ NEW

---

## Path Configuration Guide

⭐ **NEW SECTION** - Understand path requirements before you start

CharLOTTE uses paths in three different contexts. Understanding these differences prevents common configuration errors.

### Path Types Explained

#### 1. Shell Commands (Variable Expansion Works)

In bash/terminal, you can use variables:

```bash
export SCIMT_DIR=/path/to/SCIMT
cd $SCIMT_DIR/charlotte-test
bash ../Pipeline/train_SC_venv.sh config.cfg $SCIMT_DIR/venv_sound $SCIMT_DIR/venv_copper
```

✅ **Variables expand correctly**

#### 2. Shell Config Files (.cfg) (Variable Expansion Works)

Configuration files sourced by bash can use variables:

```bash
# my-config.cfg
MODULE_HOME_DIR=$SCIMT_DIR  # ✅ Works
PARALLEL_TRAIN=$BASE_DIR/data/csv/train.csv  # ✅ Works
```

✅ **Variables expand correctly**

#### 3. YAML Config Files (.yaml) (NO Variable Expansion!)

YAML parsers don't expand shell variables:

```yaml
# ❌ WRONG - Variables won't expand
save: $BASE_DIR/models/nmt_models/an-en/PRETRAIN
train_data: $BASE_DIR/data/csv/train.no_overlap_v1.csv

# ✅ CORRECT - Use absolute paths
save: /Users/yourname/charlotte-project/models/nmt_models/an-en/PRETRAIN
train_data: /Users/yourname/charlotte-project/data/csv/train.no_overlap_v1.csv
```

❌ **Variables do NOT expand - use absolute paths**

### Setup Your Environment Variables

Add these to your shell profile once:

```bash
# Set your actual paths
export SCIMT_DIR=/absolute/path/to/SCIMT  # e.g., /Users/john/projects/SCIMT
export BASE_DIR=/absolute/path/to/project  # e.g., /Users/john/charlotte-project

# Make permanent
echo "export SCIMT_DIR=$SCIMT_DIR" >> ~/.bashrc  # or ~/.zshrc for zsh
echo "export BASE_DIR=$BASE_DIR" >> ~/.bashrc
```

### Path Checklist

Before creating config files:

- [ ] Set `SCIMT_DIR` and `BASE_DIR` environment variables
- [ ] Use `$SCIMT_DIR` and `$BASE_DIR` in shell commands and .cfg files  
- [ ] Use **absolute paths only** in .yaml files (replace all variables)
- [ ] Test: `echo $SCIMT_DIR` should print your SCIMT path

---

## Prerequisites

### Python Environment
This project requires **Python 3.10 or higher** (Python 3.8 reached end-of-life in October 2024).

**Verify your Python version:**
```bash
python3 --version  # Should show 3.10.0 or higher
```

### Which Requirements File Should I Use?

CharLOTTE has **four requirements files** - here's how to choose:

| Requirements File | When to Use | Environment | Notes |
|-------------------|-------------|-------------|-------|
| **requirements-minimal.txt** | ✅ **Recommended for most users** | venv_sound | Clean, minimal dependencies with modern versions. Best for fresh installations. |
| **sound.requirements.txt** | Conda users only | conda: sound | Full conda environment export. Contains conda-specific packages. Only use if you need exact conda environment replication. |
| **copper.requirements.txt** | SC model training | venv_copper | Legacy fairseq environment with specific versions (numpy <1.24, torch 2.4.1). Required for all users. |
| **nmt.requirements.txt** | End-to-end NMT pipeline | venv_nmt | Modern PyTorch + Lightning for NMT training and translation. Required for complete pipeline. |

**Decision Tree:**

```
Are you installing venv_sound (Environment 1)?
├─ Using venv? → Use requirements-minimal.txt ✅ (recommended)
└─ Using conda? → Use sound.requirements.txt

Are you installing venv_copper (Environment 2)?
└─ Everyone uses copper.requirements.txt ✅

Are you installing venv_nmt (Environment 3)?
└─ Everyone uses nmt.requirements.txt ✅
```

**Quick Reference:**
- **venv users**: `requirements-minimal.txt` (sound) + `copper.requirements.txt` (copper) + `nmt.requirements.txt` (nmt)
- **conda users**: `sound.requirements.txt` (sound) + `copper.requirements.txt` (copper) + `nmt.requirements.txt` (nmt)

---

### Dependency Sets Explained

The project has three separate dependency sets:

1. **NMT Training (sound environment)**: PyTorch, Lightning, transformers, SentencePiece, sacrebleu, plus NLP tools
   ```bash
   pip install -r requirements-minimal.txt  # Minimal installation (recommended for venv)
   # OR
   pip install -r sound.requirements.txt    # Full installation (conda only - contains conda-specific packages)
   ```

2. **SC Model Training (copper environment)**: fairseq, CopperMT dependencies
   ```bash
   pip install -r copper.requirements.txt
   ```
   **Note:** fairseq may fail to install from PyPI on some systems. See Troubleshooting section below for git installation method.

3. **NMT Pipeline (nmt environment)**: Modern PyTorch + Lightning for end-to-end translation
   ```bash
   pip install -r nmt.requirements.txt
   ```

### External Tools
- **FastAlign**: Required for word alignment in cognate detection
  - Install: https://github.com/clab/fast_align
  - **Important for CMake 4.x users**: Update `CMakeLists.txt` line 2 from `VERSION 2.8` to `VERSION 3.5` before building
  - **Location**: Scripts expect FastAlign at `./../fast_align/build/{fast_align,atools}` relative to SCIMT root
    - Either install there, or create symlinks: `ln -s ~/bin/fast_align /path/to/SCIMT/../fast_align/build/fast_align`

---

## Installing FastAlign

FastAlign is required for word alignment in cognate detection. Install it before proceeding with CharLOTTE installation.

### Installation Steps

```bash
# Clone FastAlign
git clone https://github.com/clab/fast_align.git
cd fast_align

# For CMake 4.x users: Update CMakeLists.txt first
# Edit line 2: change "VERSION 2.8" to "VERSION 3.5"

# Build FastAlign
mkdir build
cd build
cmake ..
make

# Verify installation
./fast_align --help
```

### Setting Up FastAlign Path

CharLOTTE scripts expect FastAlign at `./../fast_align/build/` relative to the SCIMT root directory.

**Option 1: Install in expected location**
```bash
# Clone fast_align as a sibling directory to SCIMT
cd /path/to/parent/directory  # The directory containing SCIMT
git clone https://github.com/clab/fast_align.git
cd fast_align
mkdir build && cd build && cmake .. && make
```

**Option 2: Create symlinks**
```bash
# If FastAlign is installed elsewhere (e.g., ~/bin)
mkdir -p /path/to/SCIMT/../fast_align/build
ln -s ~/bin/fast_align /path/to/SCIMT/../fast_align/build/fast_align
ln -s ~/bin/atools /path/to/SCIMT/../fast_align/build/atools
```

**Verify FastAlign is accessible:**
```bash
test -f ../fast_align/build/fast_align && echo "✓ FastAlign found" || echo "✗ FastAlign not found"
```

---

## Installation

CharLOTTE requires **three separate environments** due to dependency conflicts between fairseq (used by CopperMT) and newer PyTorch versions, and to isolate NMT training dependencies.

You can use either **venv** or **conda** - both work equally well for isolating the conflicting dependencies.

### Environment 1: Sound (for NMT training and SC cognate extraction)

**Using venv (recommended):**
```bash
python3.10 -m venv venv_sound
source venv_sound/bin/activate  # On Windows: venv_sound\Scripts\activate
pip install -r requirements-minimal.txt  # Includes all necessary dependencies

# Install spaCy language models (required for SC cognate extraction)
python -m spacy download es_core_news_sm    # Spanish model
python -m spacy download xx_sent_ud_sm       # Multilingual sentence model
# Add other language models as needed for your language pairs
```

**Using conda:**
```bash
conda create -n sound python=3.10
conda activate sound
pip install -r sound.requirements.txt  # Note: contains conda-specific packages
python -m spacy download es_core_news_sm
python -m spacy download xx_sent_ud_sm
```

**Important**: For SC cognate extraction, you need these NLP tools (included in requirements-minimal.txt):
- `nltk` - Natural language tokenization
- `spacy` - Advanced tokenization for various languages
- `indic-nlp-library` - Indic language support
- `python-Levenshtein` - Edit distance calculations for cognate detection

### Environment 2: Copper (for SC model training with CopperMT)

**Using venv:**
```bash
python3.10 -m venv venv_copper  # Python 3.8 is deprecated - use 3.10+
source venv_copper/bin/activate  # On Windows: venv_copper\Scripts\activate

# Install prerequisites first
pip install cython "numpy<1.24,>=1.21" torch==2.4.1

# Install fairseq from git (PyPI version may fail on some systems)
pip install git+https://github.com/pytorch/fairseq.git@v0.10.2

# Install remaining dependencies
pip install sentencepiece pandas matplotlib scipy scikit-learn jupyter
```

**Using conda:**
```bash
conda create -n copper python=3.10  # Python 3.8 is deprecated
conda activate copper
pip install -r copper.requirements.txt
```

### Environment 3: NMT (for end-to-end NMT model training and translation)

**Using venv:**
```bash
python3.10 -m venv venv_nmt
source venv_nmt/bin/activate  # On Windows: venv_nmt\Scripts\activate
pip install -r nmt.requirements.txt
```

**Using conda:**
```bash
conda create -n nmt python=3.10
conda activate nmt
pip install -r nmt.requirements.txt
```

**What venv_nmt is used for:**
- Training transformer-based NMT models (BART architecture)
- Running end-to-end translation pipeline (Portuguese→English)
- Generating translations and computing BLEU/chrF scores
- Full CharLOTTE quickstart demonstration (5 phases)

**Key dependencies:**
- `transformers` - BART model architecture
- `lightning` - PyTorch Lightning training framework
- `sentencepiece` - Tokenization (same as venv_sound)
- `sacrebleu` - Translation evaluation metrics
- `torch` - Modern PyTorch (compatible with Lightning)

### Why three environments?
- `sound` environment: SC cognate extraction, tokenizer training
- `copper` environment: SC model training with fairseq (requires PyTorch 2.4.1 and numpy <1.24)
- `nmt` environment: NMT model training and translation (uses modern PyTorch and Lightning)
- All three are **required** for the complete CharLOTTE pipeline
- Separate environments prevent version conflicts between fairseq (copper) and modern PyTorch (nmt)

### Important Notes:
- Python 3.8 reached end-of-life (EOL) in October 2024 - use Python 3.10+ instead
- fairseq from PyPI may fail with permission errors on macOS - use git installation method shown above
- numpy 1.19 (specified in copper.requirements.txt) is incompatible with Python 3.10 - use numpy 1.21-1.23 instead

### Using `train_SC.sh` with venv

The original `train_SC.sh` uses `conda activate` commands. We provide two options:

**Option A: Use `train_SC_venv.sh`** (venv-compatible fork):
```bash
bash Pipeline/train_SC_venv.sh <config_file> <venv_sound_path> <venv_copper_path>

# Example:
bash Pipeline/train_SC_venv.sh Pipeline/cfg/SC/my_config.cfg ./venv_sound ./venv_copper
```

**Option B: Use original `train_SC.sh`** with conda:
```bash
conda activate sound  # or copper, depending on stage
bash Pipeline/train_SC.sh <config_file>
```

---

## Installing CopperMT

**CopperMT** is the framework that trains SC (Sound Correspondence) models for cognate prediction.

### Prerequisites for Moses SMT

Before installing CopperMT, ensure you have:

- **Boost** (≥1.64): Required for Moses decoder
  - macOS: `brew install boost`
  - Ubuntu/Debian: `sudo apt install libboost-all-dev`
- **Additional packages**: See [Moses installation docs](https://www.statmt.org/moses/?n=Development.GetStarted) for distribution-specific requirements

### Clone and Setup CopperMT

From the SCIMT root directory:

```bash
cd CopperMT
git clone https://github.com/clefourrier/CopperMT.git
cd CopperMT

# Initialize git submodules (Moses decoder and mgiza for SMT models)
git submodule init
git submodule update

# Install mgiza
cd submodules/mgiza/mgizapp
cmake .
make
make install
cp scripts/merge_alignment.py bin/

# Install Moses decoder
cd ../../mosesdecoder
# Note: On macOS, you may need to checkout branch clang-error first
bjam -j4 -q -d2

# Copy custom scripts
cd ../../../CopperMTfiles
python3 move_files.py
```

This clones the CopperMT repository, installs Moses decoder and mgiza for SMT models, and copies custom scripts into it.

### Verify CopperMT Installation

⭐ **NEW SECTION** - Verify before proceeding to Quick Test

Run these commands to ensure CopperMT is correctly installed:

```bash
# Check Moses scripts exist
test -f $SCIMT_DIR/CopperMT/CopperMT/submodules/mosesdecoder/scripts/training/clean-corpus-n.perl && \
  echo "✓ Moses scripts found" || echo "✗ Moses scripts missing"

# Check Moses binaries built
test -f $SCIMT_DIR/CopperMT/CopperMT/submodules/mosesdecoder/bin/moses && \
  echo "✓ Moses binary built" || echo "✗ Moses binary missing - bjam build failed"

# Check mgiza built
test -f $SCIMT_DIR/CopperMT/CopperMT/submodules/mgiza/mgizapp/bin/mgiza && \
  echo "✓ mgiza built" || echo "✗ mgiza missing"
```

If any checks fail, see [../TROUBLESHOOTING.md](TROUBLESHOOTING.md#moses-decoder-not-installed).

---

## Pre-Flight Check

⭐ **NEW SECTION** - Verify everything before Quick Test

Before running the Quick Test (~5 minutes), verify all prerequisites are met. This saves time by catching issues early.

### Run Pre-Flight Checks

```bash
echo "=== CharLOTTE Pre-Flight Check ==="

# 1. Check virtual environments exist
echo "1. Checking virtual environments..."
test -d $SCIMT_DIR/venv_sound && echo "  ✓ venv_sound exists" || echo "  ✗ venv_sound missing - re-run installation"
test -d $SCIMT_DIR/venv_copper && echo "  ✓ venv_copper exists" || echo "  ✗ venv_copper missing - re-run installation"
test -d $SCIMT_DIR/venv_nmt && echo "  ✓ venv_nmt exists" || echo "  ✗ venv_nmt missing - re-run installation"

# 2. Check fairseq installation
echo "2. Checking fairseq in venv_copper..."
source $SCIMT_DIR/venv_copper/bin/activate
python -c "import fairseq; print(f'  ✓ fairseq {fairseq.__version__}')" 2>/dev/null || echo "  ✗ fairseq not installed"
deactivate

# 3. Check FastAlign
echo "3. Checking FastAlign..."
which fast_align >/dev/null && echo "  ✓ FastAlign found at $(which fast_align)" || echo "  ✗ FastAlign not in PATH"

# 4. Check test config exists
echo "4. Checking Quick Test config..."
test -f $SCIMT_DIR/charlotte-test/test-sc-es-pt.cfg && echo "  ✓ Test config exists" || echo "  ✗ Test config missing"

# 5. Check CopperMT/Moses
echo "5. Checking CopperMT/Moses..."
test -f $SCIMT_DIR/CopperMT/CopperMT/submodules/mosesdecoder/bin/moses && echo "  ✓ Moses built" || echo "  ✗ Moses not built"

echo ""
echo "=== Pre-Flight Check Complete ==="
echo "If all checks passed (✓), proceed to Quick Test."
echo "If any checks failed (✗), fix them before continuing."
```

### What Each Check Means

| Check | What It Verifies | If It Fails |
|-------|------------------|-------------|
| venv_sound | Sound environment exists | Re-run installation - Environment 1 (Sound) |
| venv_copper | Copper environment exists | Re-run installation - Environment 2 (Copper) |
| venv_nmt | NMT environment exists | Re-run installation - Environment 3 (NMT) |
| fairseq | fairseq installed in copper | Install fairseq: `pip install git+https://github.com/pytorch/fairseq.git@v0.10.2` |
| FastAlign | FastAlign in PATH | See [Installing FastAlign](#installing-fastalign) |
| Test config | Quick Test config file | Config ships with repository - check git clone |
| Moses | Moses decoder built | Re-run CopperMT installation, check build logs |

---

## Quick Test

### Verifying Your Setup (~5 minutes)

Before running the full pipeline, test your setup with a minimal RNN-based SC model using Spanish-Portuguese parallel data. This test verifies all components including fairseq and RNN training.

### Prerequisites

Ensure you have:
- Completed installation steps (both environments)
- Created separate virtual environments (venv_sound and venv_copper)
- Installed CopperMT with Moses decoder

### Test RNN-based SC Training

```bash
# Navigate to your test directory
cd $SCIMT_DIR
mkdir -p charlotte-test
cd charlotte-test

# Copy the provided test configuration
cp $SCIMT_DIR/charlotte-test/test-sc-es-pt.cfg ./test-sc-es-pt.cfg

# Optional: View the configuration
cat test-sc-es-pt.cfg
# This config uses:
# - Spanish (es) → Portuguese (pt) translation
# - RNN model type (fairseq-based BiGRU with Luong attention)
# - 1000 cognate pairs for training
# - 20 epochs (completes in ~4 minutes)

# Run SC training with separate virtual environments
bash ../Pipeline/train_SC_venv.sh test-sc-es-pt.cfg ../venv_sound ../venv_copper 2>&1 | tee test_run_es_pt.log
```

**What This Tests**:
- ✅ FastAlign cognate detection (Phase 1-2)
- ✅ fairseq RNN model training (Phase 3)
- ✅ Python 3.10 compatibility patches
- ✅ Virtual environment isolation
- ✅ Model checkpoint saving
- ✅ Training metrics logging

**Expected Timeline**:
- Phase 1-2 (Cognate Detection): ~30 seconds
- Phase 3 (RNN Training): ~4 minutes (20 epochs)
- Total: ~5 minutes

**Success Indicators**:
```
✓ Cognate pairs extracted and saved to charlotte-test/cognates_es_pt_RNN-default_S-1000/
✓ Training progresses through 20 epochs
✓ Validation loss decreases (e.g., 2.779 → 1.084)
✓ Checkpoints saved to charlotte-test/sc_models_es_pt/.../checkpoints/
✓ Best checkpoint identified (typically checkpoint15.pt with lowest validation loss)
✓ Log shows: "done training in X seconds"
```

**Monitoring Progress**:
```bash
# In another terminal, watch training progress
tail -f test_run_es_pt.log | grep -E "epoch|loss|ppl"
```

**Expected Training Metrics**:
| Epoch | Train Loss | Val Loss | Val PPL | Notes |
|-------|------------|----------|---------|-------|
| 1     | 4.023      | 2.779    | 6.86    | Initial |
| 5     | 1.631      | 1.414    | 2.67    | Rapid improvement |
| 11    | 1.099      | 1.123    | 2.18    | Converging |
| 15    | 0.963      | 1.084    | 2.12    | Best checkpoint |
| 20    | 0.869      | 1.123    | 2.17    | Final |

**If This Works**:
- Your environment is correctly set up for RNN-based SC models
- You can proceed to full-scale training
- The trained model is ready for cognate translation (es→pt)

**If This Fails**, check:

1. **Virtual Environment Issues**:
   ```bash
   # Verify venv_sound exists
   ls $SCIMT_DIR/venv_sound/bin/activate

   # Verify venv_copper exists
   ls $SCIMT_DIR/venv_copper/bin/activate

   # Check fairseq installation
   source $SCIMT_DIR/venv_copper/bin/activate
   pip show fairseq
   python -c "import fairseq; print(fairseq.__version__)"
   ```

2. **FastAlign Path**:
   ```bash
   # Should be in venv_sound
   source $SCIMT_DIR/venv_sound/bin/activate
   which fast_align
   ```

3. **Python 3.10 Compatibility**:
   - The RNN training requires patches for fairseq 0.10.2
   - These are automatically applied during setup
   - Check `venv_copper/lib/python3.10/site-packages/fairseq/dataclass/configs.py`

4. **Dependencies**:
   ```bash
   source $SCIMT_DIR/venv_copper/bin/activate
   pip list | grep -E "(torch|fairseq|sacrebleu)"
   # Expected versions:
   # fairseq==0.10.2
   # sacrebleu==1.5.1
   # torch>=2.0
   ```

5. **Log File Analysis**:
   ```bash
   # Check for specific errors
   grep -i "error\|traceback\|failed" test_run_es_pt.log

   # Verify training started
   grep "begin training epoch" test_run_es_pt.log

   # Check final status
   tail -50 test_run_es_pt.log
   ```

**Common Issues**:

- **`Optional[str]` type error**: fairseq 0.10.2 incompatibility with Python 3.10
  - Solution: Re-run the setup script which applies necessary patches

- **`cannot import name 'TOKENIZERS'`**: Wrong sacrebleu version
  - Solution: `pip install 'sacrebleu==1.5.1'` in venv_copper

- **Memory errors**: Insufficient RAM for fairseq
  - Solution: Reduce batch size in test-sc-es-pt.cfg (currently 10)

**Next Steps After Success**:
- Review the trained model checkpoints
- Proceed to full-scale SC training with larger datasets

---

## Installation Verification

⭐ **NEW SECTION** - Final verification checklist

After completing installation, verify everything works:

```bash
# Test core imports
cd $SCIMT_DIR
python test_imports.py
```

Expected output:
```
Testing core imports...
✓ PyTorch 2.8.0
✓ Lightning 2.5.5
✓ Transformers 4.57.0
✓ SentencePiece
✓ NMT modules

✓ All core dependencies are installed correctly!
```

### Verification Checklist

- [ ] Pre-flight check passes (all ✓)
- [ ] `python test_imports.py` succeeds
- [ ] Quick Test completes successfully (~5 min)
- [ ] Training metrics match expected values
- [ ] Checkpoints saved to `charlotte-test/sc_models_es_pt/.../checkpoints/`

If all items pass, you're ready for full experiments → [../EXPERIMENTATION.md](EXPERIMENTATION.md)

---

## Next Steps

✅ Installation complete!  
✅ Quick Test passed!

**What's next?**

1. **Run full experiments** → [../EXPERIMENTATION.md](EXPERIMENTATION.md)
2. **Encountered issues?** → [../TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. **Understand the pipeline** → Read the end-to-end example in EXPERIMENTATION.md

---

**[← Back to README](../README.md)** | **[Experimentation Guide →](EXPERIMENTATION.md)** | **[Troubleshooting →](TROUBLESHOOTING.md)**
