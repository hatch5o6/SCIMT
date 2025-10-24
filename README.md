# CharLOTTE

**Char**acter-**L**evel **O**rthographic **T**ransfer for **T**oken **E**mbeddings

This is the codebase for **CharLOTTE**, a system that leverages character correspondences between related languages in low-resource NMT.

The CharLOTTE system assumes that the phenomenon of systematic sound correspondence in linguistics is reflected in character correspondences in orthography. For example, *j-lh* and *h-f* correspondences between Spanish and Portuguese, seen in word pairs:
- *ojo, olho*
- *ajo, alho*
- *hierro, ferro*
- *horno, forno*
- *hijo, filho*

CharLOTTE learns these character correspondences with what we call **SC models** and trains tokenizers and NMT models that exploit them so as to increase vocabulary overlap between related high and low-resource languages. CharLOTTE utilizes a **language-agnostic approach**, requiring only the NMT parallel training, validation, and testing data; though additional sets of known language-specific cognates can also be provided.

## What are SC Models?

**SC** stands for **Sound Correspondence** (though more accurately, "character correspondence" since the system operates on orthography rather than phonetic transcriptions).

SC models learn systematic character-level mappings between related languages. CharLOTTE uses these models to address a fundamental challenge in low-resource NMT:

**The Challenge**: Training high-quality NMT requires large parallel datasets, but low-resource languages have limited data.

**The SC Solution**:
1. Identify a high-resource language related to your low-resource target (e.g., Spanish for Portuguese, French for Mauritian Creole)
2. Train an **SC model** to learn character correspondences between the high-resource and low-resource languages
3. **Apply the SC model** to transform high-resource parallel data, making it orthographically similar to the low-resource language
4. Train NMT using both the original low-resource data AND the SC-normalized high-resource data

**Example**: For Portuguese→English NMT with limited Portuguese-English data:
- Train SC model: Spanish → Portuguese character correspondences
- Apply to data: Transform Spanish-English corpus to look like Portuguese-English
- Result: Spanish word *hijo* → Portuguese-like *filho* (learning correspondences like *j→lh*, *o→o*)
- Train NMT with augmented data, benefiting from increased vocabulary overlap

**SC Model Types**:
- **RNN**: Sequence-to-sequence neural model for cognate prediction
- **SMT**: Statistical machine translation model for cognate prediction

Both are trained using the CopperMT framework and can predict character-level transformations to generate plausible cognates.

**Key Terms**:
- **SC Model** (Sound Correspondence Model): Character-level seq2seq model that learns systematic mappings between related languages
- **Cognate**: Words in different languages derived from the same origin (e.g., Spanish *noche* / Portuguese *noite*)
- **Data Augmentation**: Transforming high-resource data to augment low-resource training sets

## The Three Models in CharLOTTE

CharLOTTE trains **three complementary models** that work together to enable low-resource NMT:

### 1. SC Model (Sound Correspondence Model)
**Type**: Character-level sequence-to-sequence model (RNN or SMT)
**Purpose**: Learns phonetic/orthographic transformations between related languages
**Training data**: Cognate pairs extracted from parallel corpora (e.g., Spanish-Portuguese word pairs)
**What it learns**: Character-level mappings like *j→lh*, *ción→ção*, *h→f*
**Example**: Transforms Spanish *hijo* → Portuguese-like *filho*
**Output**: Model that can transform high-resource text to look like low-resource language

### 2. Tokenizer (SentencePiece)
**Type**: Statistical subword segmentation model (not neural)
**Purpose**: Creates a shared vocabulary that works across multiple languages
**Training data**: Combined text from SC-transformed high-resource + native low-resource + target language
**What it learns**: How to segment text into subword units (tokens) optimally across all languages
**Example**: Learns that *-tion* (English), *-ção* (Portuguese), and SC-transformed *-ción→ção* share subword patterns
**Output**: Unified tokenizer that maximizes vocabulary overlap between related languages

### 3. NMT Model (Neural Machine Translation)
**Type**: Transformer encoder-decoder (BART-based)
**Purpose**: Performs the actual translation from low-resource to target language
**Training data**: Original low-resource parallel data + SC-augmented high-resource data
**What it learns**: Semantic mappings for translation, benefiting from increased training data and vocabulary overlap
**Example**: Translates Portuguese → English using both native Portuguese-English pairs and SC-transformed Spanish-English pairs
**Output**: Production translation model with improved quality due to data augmentation

### How They Work Together

```
Spanish text ("hijo habla español")
        ↓
   [SC Model] ← learns j→lh, h→f character rules
        ↓
Synthetic Portuguese-like text ("filho fala español")
        ↓
   [Tokenizer] ← learns shared subwords across es/pt/en
        ↓
Subword tokens (['▁fil', 'ho', '▁fal', 'a', ...])
        ↓
   [NMT Model] ← translates using augmented training data
        ↓
English translation ("son speaks Spanish")
```

**Key Insight**: The SC model creates synthetic training data that looks like the low-resource language, enabling the NMT model to learn from related high-resource data. The shared tokenizer ensures maximum vocabulary overlap across all languages.

## Quick Start

CharLOTTE provides an end-to-end pipeline for SC-augmented NMT training. Follow the installation steps below, then verify your setup with the full quickstart test before running experiments.

### The CharLOTTE Pipeline

Here's the complete workflow from data to trained NMT model:

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: PREPARE DATA (30-60 min - your effort)           │
├─────────────────────────────────────────────────────────────┤
│  • Collect parallel corpora (low-resource + high-resource)  │
│  • Create train/val/test splits with no overlap             │
│  • Prepare CSV configuration files                          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: TRAIN SC MODEL (1-3 hours - automated)           │
├─────────────────────────────────────────────────────────────┤
│  • Extract cognates using FastAlign                         │
│  • Train SC model (RNN or SMT) on cognate pairs             │
│  • Learn character correspondences                          │
│  Output: SC model that transforms high→low resource text    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: APPLY SC MODEL (10-20 min - automated)           │
├─────────────────────────────────────────────────────────────┤
│  • Transform high-resource data to look like low-resource   │
│  • Create augmented training data                           │
│  Output: Expanded dataset with increased vocabulary overlap │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4: TRAIN TOKENIZER (10-30 min - automated)          │
├─────────────────────────────────────────────────────────────┤
│  • Train SentencePiece BPE tokenizer on combined data       │
│  • Create shared vocabulary across language pairs           │
│  Output: Tokenizer model (.model + .vocab files)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 5: TRAIN NMT MODEL (6-48 hours - automated)         │
├─────────────────────────────────────────────────────────────┤
│  • Train transformer NMT on augmented data                  │
│  • Validate on held-out validation set                      │
│  • Save best checkpoint based on validation BLEU            │
│  Output: Trained NMT model checkpoint                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  PHASE 6: EVALUATE (15-30 min - automated)                 │
├─────────────────────────────────────────────────────────────┤
│  • Run inference on test set                                │
│  • Calculate BLEU, chrF, TER metrics                        │
│  Output: Translation quality scores                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Points**:
- **Total time**: 8-50 hours (mostly unattended training)
- **Manual effort**: ~1-2 hours (data preparation and configuration)
- **GPU recommended**: Phases 2 and 5 are 8-10x faster with GPU
- **Checkpoint resume**: All training phases support resuming from interruptions

**Quickstart Test**: After installation, run the automated quickstart test (~30-45 minutes) to validate your complete setup by running all 6 phases from SC training through NMT translation.

### Installation Checklist

Follow these steps in order:

- [ ] **1. Install Prerequisites** (~10 min)
  - Python 3.10+, Git, CMake, Boost
  - → [docs/SETUP.md](docs/SETUP.md#prerequisites)

- [ ] **2. Clone Repository** (1 min)
  ```bash
  git clone https://github.com/hatch5o6/SCIMT.git
  cd SCIMT
  ```

- [ ] **3. Install FastAlign** (~5 min)
  - Required for cognate extraction
  - → [docs/SETUP.md](docs/SETUP.md#installing-fastalign)

- [ ] **4. Create Virtual Environments** (~30 min)
  - Three environments required (sound + copper + nmt)
  - → [docs/SETUP.md](docs/SETUP.md#installation)

- [ ] **5. Install CopperMT** (~30-60 min)
  - Includes Moses decoder build
  - → [docs/SETUP.md](docs/SETUP.md#installing-coppermt)

- [ ] **6. Run Quickstart Test** (~30-45 min)
  - Validates complete 6-phase pipeline (SC→Tokenizer→NMT→Translation)
  - → [docs/QUICKSTART.md](docs/QUICKSTART.md) or [docs/SETUP.md](docs/SETUP.md#quickstart-test-full-end-to-end-pipeline-validation)

- [ ] **8. Start Full Experiments**
  - → [docs/EXPERIMENTATION.md](docs/EXPERIMENTATION.md)

**Total Time**: 1-2 hours (mostly waiting for builds)

## Documentation

**Getting Started (in order)**:
1. **[docs/SETUP.md](docs/SETUP.md)** - Installation and quickstart test ← **START HERE**
2. **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Full pipeline test with Spanish→Portuguese→English
3. **[docs/EXPERIMENTATION.md](docs/EXPERIMENTATION.md)** - Complete Portuguese→English workflow

**Reference Documentation** (as needed):

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md)** | Obtaining and formatting training data | Before preparing your own data |
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | Complete parameter reference for all config types | When configuring experiments |
| **[docs/MONITORING.md](docs/MONITORING.md)** | Training monitoring and evaluation | During and after training |
| **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** | Common issues and solutions | When you encounter errors |

## Key Features

- **Language-agnostic**: Works with any related language pair
- **Two SC model types**: RNN (neural) or SMT (statistical)
- **Flexible augmentation**: Use SC models with any NMT architecture
- **Automatic cognate extraction**: Uses FastAlign to find related words
- **End-to-end pipeline**: From parallel data to trained NMT models
- **Validation included**: Automated quickstart test validates complete pipeline

## What Makes Languages "Related"?

CharLOTTE works best when languages share systematic character correspondences. Your language pair is likely suitable if:

- **Same language family**: Romance (Spanish/Portuguese), Germanic (English/German), Slavic (Russian/Polish), etc.
- **Significant borrowing/contact**: English-French, Hindi-Urdu, Turkish-Arabic (loanwords)
- **Test**: Can you find 20+ word pairs with similar spellings? (*hijo/filho*, *noche/noite*, *padre/father*)

**When SC augmentation helps most**:
- Related languages from same family (e.g., Spanish→Portuguese, French→Mauritian Creole)
- High-resource language has 100k+ parallel sentences
- Low-resource language has < 50k parallel sentences

**When SC augmentation may not help**:
- Unrelated languages (e.g., English-Japanese, Arabic-Chinese)
- Different writing systems (Latin vs. Cyrillic vs. logographic)
- Already have 100k+ low-resource sentences (diminishing returns)

## Requirements

**System Requirements**:
- Python 3.10 or higher
- PyTorch 2.0+
- fairseq 0.10.2 (for SC RNN models)
- FastAlign (for cognate detection)
- Moses decoder (for SC SMT models)
- **GPU**: 8GB+ VRAM recommended (NVIDIA GPU with CUDA support)
- **CPU**: Training supported but 8-10x slower; minimum 8 GB RAM
- **Disk space**: 20 GB recommended
  - SCIMT installation: ~2 GB
  - SC models: ~500 MB per model
  - NMT models: ~2-5 GB per model (includes checkpoints)
  - Training data: 100 MB - 1 GB (varies)
- **OS**: Linux (Ubuntu/Debian), macOS, or Windows (WSL2)

## Quick Commands

```bash
# Verify Python version
python3 --version  # Should be 3.10+

# Check if FastAlign is installed
which fast_align

# Run Quickstart Test (after setup)
cd charlotte-test
./run_full_quickstart.sh

# Train full SC model
bash Pipeline/train_SC.sh configs/sc/my-config.cfg

# Train NMT model
cd NMT
python train.py -c configs/nmt/my-config.yaml -m TRAIN
```

## Project Structure

```
SCIMT/
├── README.md                    # This file
├── docs/                        # Detailed documentation
│   ├── SETUP.md                # Installation and quickstart test
│   ├── QUICKSTART.md           # Full pipeline test guide
│   ├── EXPERIMENTATION.md      # Complete workflow guide
│   ├── DATA_PREPARATION.md     # Data preparation guide
│   ├── CONFIGURATION.md        # Parameter reference
│   ├── MONITORING.md           # Training monitoring & evaluation
│   └── TROUBLESHOOTING.md      # Common issues
├── Pipeline/                    # SC training scripts
├── NMT/                         # NMT training code
├── CopperMT/                    # SC model framework
├── charlotte-test/              # Quickstart test files
└── requirements-minimal.txt     # Python dependencies
```

## Getting Help

- **Installation issues**: See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)
- **Bug reports**: Open an issue on GitHub
- **Questions**: Check documentation or open a discussion

## Citation

If you use CharLOTTE in your research, please cite:

```bibtex
@inproceedings{charlotte2024,
  title={CharLOTTE: Character-Level Orthographic Transfer for Token Embeddings},
  author={Your Name and Collaborators},
  booktitle={Proceedings of Conference},
  year={2024}
}
```

## License

[Add your license information here]

---

**Ready to get started?** → [docs/SETUP.md](docs/SETUP.md)
