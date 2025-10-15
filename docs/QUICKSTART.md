# CharLOTTE Quick Start

**Purpose:** Run your first end-to-end CharLOTTE experiment - from SC model training through actual Portuguese→English translation.

---

**🎯 Who is this for?**
- **First-time users** who want to see CharLOTTE work end-to-end in 30-45 minutes
- Users who prefer **automated scripts** over manual step-by-step configuration
- Anyone wanting **quick validation** that the system works before diving into experiments

**🎯 What you'll do:**
- Run ONE command (`run_full_quickstart.sh`) that executes all phases automatically
- Watch the complete pipeline from SC training → tokenization → NMT → translation → baseline comparison
- Get actual English translations from Portuguese with automatic quality comparison
- View training loss curves for both SC and NMT models

**🎯 What's next:**
- After success, move to [EXPERIMENTATION.md](EXPERIMENTATION.md) to learn manual control for your own language pairs

---

This guide demonstrates the complete CharLOTTE workflow with a small Spanish→Portuguese→English example, taking you from character correspondence learning all the way to generating and evaluating translations.

**Prerequisites**: Complete [SETUP.md](SETUP.md) installation and verify all components are working (all three virtual environments: venv_sound, venv_copper, venv_nmt).

---

## What You'll Accomplish

Run the complete CharLOTTE pipeline with **baseline comparison**, building **three distinct models** and comparing against a non-augmented baseline:

### SC-Augmented Pipeline (Phases 1-5)

1. **Prepare data** - Download FLORES-200 dataset (1600 train, 200 val, 200 test sentences)
2. **Train SC model** - Learn Spanish→Portuguese character correspondences
   - **Model 1: `es_pt_RNN-default_S-1000`** - Character-level RNN trained on cognate pairs
   - Learns mappings like *j→lh*, *h→f*, *ción→ção*
   - Saved to: `sc_models_es_pt/es_pt_RNN-default_S-1000/`
   - **NEW: Automatic loss curve visualization**
3. **Apply SC model** - Transform Spanish data to augment Portuguese training data
   - Creates synthetic Portuguese-like text from Spanish corpus
4. **Train tokenizer** - Build multilingual vocabulary with SC-augmented data
   - **Model 2: `SC_es2pt-pt_en.model`** - SentencePiece tokenizer (8000 vocab)
   - Learns shared subword vocabulary across SC-transformed Spanish, Portuguese, and English
   - Saved to: `spm_models/SC_es2pt-pt_en/`
5. **Train NMT model** - Train Portuguese→English translator with augmentation
   - **Model 3: `pt-en_TRIAL_s=1000`** - BART-based transformer (9.6M parameters)
   - 4-layer encoder/decoder, trained on augmented data (3200 sentences: 1600 pt-en + 1600 SC-transformed es-en)
   - Saved to: `nmt_models/pt-en_TRIAL_s=1000/checkpoints/`
   - **NEW: Automatic loss curve visualization**
6. **Evaluate & translate** - Generate actual English translations and evaluate quality

### Baseline Pipeline (Phases B1-B3) - NEW!

7. **Train baseline tokenizer** - Portuguese-English only (no SC data)
   - SentencePiece tokenizer on pt-en vocabulary only
8. **Train baseline NMT** - Portuguese→English WITHOUT augmentation
   - Same architecture (9.6M parameters), trained on 1600 pt-en sentences only
9. **Evaluate baseline** - Generate translations for fair comparison

### Comparison Summary - NEW!

10. **Side-by-side BLEU scores** - See the impact of SC augmentation
    - SC-Augmented: ~38% BLEU (3200 training pairs with augmentation)
    - Baseline: ~33% BLEU (1600 training pairs, no augmentation)
    - **~16% relative improvement** from SC augmentation

**Time**: ~45-60 minutes (mostly automated, includes baseline training)
**Hardware**: GPU recommended (8-10x faster), CPU supported

**What you'll learn**: How three complementary models work together - the SC model creates synthetic Portuguese-like data from Spanish, the tokenizer creates a shared vocabulary across all languages, and the NMT model performs translation using the augmented training data.

---

## Full Pipeline Test: End-to-End Translation

This is the complete CharLOTTE workflow from SC training through actual translation. You'll see how character-level transformations enable data augmentation for low-resource NMT.

**What happens**: We train a Spanish→Portuguese SC model, use it to augment Portuguese training data, then train a Portuguese→English NMT model that benefits from the augmented data.

### Step 1: Navigate to Test Directory

```bash
cd $SCIMT_DIR/charlotte-test
```

**Note**: If `$SCIMT_DIR` is not set, use the absolute path to your SCIMT directory:
```bash
export SCIMT_DIR=/path/to/your/SCIMT/clone
cd $SCIMT_DIR/charlotte-test
```

### Step 2: Run the Full Pipeline

Execute the automated end-to-end pipeline script:

```bash
./run_full_quickstart.sh
```

The script will automatically:
- Download FLORES-200 data if needed (1600 train, 200 val, 200 test sentences)
- Run all SC-augmented phases (1-5) sequentially
- Run baseline comparison phases (B1-B3)
- Skip already-completed phases
- Generate loss curve visualizations for SC and NMT models
- Save detailed logs for each phase
- Generate translations from both SC-augmented and baseline models
- Display side-by-side BLEU score comparison

### Step 3: Monitor Progress

Watch the terminal output as the pipeline executes. You'll see each phase complete:

**Phase 1: SC Training** (~5-7 minutes)
```
==========================================
PHASE 1: Training SC Model (es->pt)
==========================================
Extracting cognates using FastAlign...
Training RNN SC model with fairseq...
BLEU: 68.32, chrF: 61.52
✓ Phase 1 Complete: SC model trained

Generating SC model loss curves...
✓ SC model loss curves saved to: logs/quickstart_phase1_sc_training_*_loss_curves.png
```

**Phase 2: SC Application** (~1 minute)
```
==========================================
PHASE 2: Applying SC Model
==========================================
Transforming Spanish text to Portuguese-like text...
✓ Phase 2 Complete: SC transformations applied
```

**Phase 3: Tokenizer Training** (~2-3 minutes)
```
==========================================
PHASE 3: Training Tokenizer
==========================================
Training multilingual SentencePiece tokenizer on:
  - SC-transformed Spanish (es2pt)
  - Portuguese (pt)
  - English (en)
✓ Phase 3 Complete: Tokenizer trained
```

**Phase 4: NMT Training** (~15-20 minutes)
```
==========================================
PHASE 4: Training NMT Model (pt->en)
==========================================
Training transformer NMT model for 500 steps...
Epoch 1: loss=4.23, val_loss=3.87
...
✓ Phase 4 Complete: NMT model trained
```

**Phase 5: Translation** (~2 minutes)
```
==========================================
PHASE 5: Generating Translations
==========================================
Translating test set (200 sentences)...
BLEU: 38.1, chrF: 14.58
✓ Phase 5 Complete: Translations generated
```

**Baseline Phases** (~20-25 minutes) - NEW!
```
==========================================
BASELINE: Training WITHOUT SC Augmentation
==========================================

PHASE B1: Training Baseline Tokenizer (pt-en only)
✓ Baseline tokenizer trained

PHASE B2: Training Baseline NMT Model (pt->en, NO augmentation)
Training on 1600 pt-en sentences only...
Epoch 1: loss=7.12, val_loss=4.56
...
✓ Baseline NMT model trained

PHASE B3: Evaluating Baseline Model
BLEU: 32.9, chrF: 14.52
✓ Baseline translations generated

==========================================
COMPARISON: SC-Augmented vs Baseline
==========================================

SC-Augmented Model (with Spanish data via SC transformations):
    "BLEU": 0.38 (38%)

Baseline Model (Portuguese-English only, NO augmentation):
    "BLEU": 0.33 (33%)

Dataset sizes:
  - SC-Augmented: 3200 sentence pairs (1600 pt-en + 1600 SC-transformed es-en)
  - Baseline: 1600 sentence pairs (pt-en only)

** ~16% BLEU improvement from SC augmentation **
```

### Step 4: Verify Results

Check the generated translations and loss curves:
```bash
# View SC-augmented translations (Portuguese → English)
head -5 nmt_models/pt-en_TRIAL_s=1000/predictions/*/predictions.txt

# View baseline translations
head -5 nmt_models/pt-en_BASELINE_s=1000_TRIAL_s=1000/predictions/*/predictions.txt

# Compare with reference translations
head -5 data/raw/test.pt-en.en

# View loss curves
open logs/quickstart_phase1_sc_training_*_loss_curves.png  # SC model
open nmt_models/pt-en_TRIAL_s=1000/logs/lightning_logs/version_0/loss_curves.png  # NMT model
```

**Expected Results**:
- **SC-Augmented BLEU**: 35-40% (with 3200 augmented training pairs)
- **Baseline BLEU**: 30-35% (with 1600 pt-en pairs only)
- **Improvement**: ~16% relative BLEU gain from SC augmentation

All outputs are saved in:

**SC-Augmented Model:**
- `sc_models_es_pt/` - Trained SC model
- `spm_models/SC_es2pt-pt_en/` - Multilingual tokenizer (SC-Spanish, Portuguese, English)
- `nmt_models/pt-en_TRIAL_s=1000/` - NMT model trained with SC augmentation
- `logs/quickstart_phase1_sc_training_*_loss_curves.png` - **SC model loss curves (NEW!)**
- `nmt_models/pt-en_TRIAL_s=1000/logs/.../loss_curves.png` - **NMT loss curves (NEW!)**

**Baseline Model:**
- `spm_models/pt_en/` - Bilingual tokenizer (Portuguese-English only)
- `nmt_models/pt-en_BASELINE_s=1000_TRIAL_s=1000/` - NMT model without augmentation

**Logs:**
- `logs/quickstart_phase*.log` - Detailed logs for each phase (both SC-augmented and baseline)

### Step 5: Understand Your Results

**SC Model Quality** (Phase 1):
- BLEU > 60: ✅ Excellent - strong character correspondences learned
- BLEU 40-60: ✅ Good - captures most systematic patterns
- BLEU < 40: ⚠️ Check if languages are actually related

**NMT Translation Quality** (Phase 5):
- **SC-Augmented BLEU 35-40**: ✅ Expected with augmentation (3200 effective training pairs)
- **Baseline BLEU 30-35**: ✅ Expected without augmentation (1600 training pairs)
- **Improvement ~16%**: ✅ Demonstrates SC augmentation value

**Loss Curves** (NEW!):
- **SC Model**: Should show smooth convergence (loss dropping from ~3.7 to ~0.8, perplexity from ~12 to ~2)
- **NMT Model**: May show validation jitter due to small validation set (200 samples), but training loss should decrease steadily

**Key Takeaways**:
1. **SC augmentation works**: Even with limited data (1600→3200 sentences via augmentation), you get measurable BLEU improvements
2. **Baseline comparison is automatic**: The quickstart now demonstrates value by comparing against a non-augmented baseline
3. **Visual feedback**: Loss curves let you monitor both SC and NMT training progress
4. **This is a demo**: With such small datasets, both models produce poor translations. Full experiments with 10k-50k sentences achieve production-quality results (BLEU 30-45+)

---

## Troubleshooting

### Issue: "venv_nmt not found"
**Solution**: The full pipeline requires all three virtual environments. If you haven't created venv_nmt yet:
```bash
cd $SCIMT_DIR
python3.10 -m venv venv_nmt
source venv_nmt/bin/activate
pip install -r nmt.requirements.txt
```

### Issue: "fast_align: command not found"
**Solution**: FastAlign not in PATH. See [SETUP.md - Installing FastAlign](SETUP.md#installing-fastalign).

### Issue: Phase 4 (NMT Training) fails or takes too long
**Solution**:
- Check GPU availability: `nvidia-smi`
- On CPU, training takes ~2-3 hours (vs. 15-20 min on GPU)
- Reduce training steps in `test-nmt-pt-en.yaml` if needed

### Issue: Low BLEU scores in Phase 5
**Possible causes**:
- This is expected for small-scale test (800 training sentences)
- BLEU 15-30 is reasonable for quickstart
- Full experiments with more data achieve 30-45 BLEU

For more detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## What's Next?

Now that you've run the complete CharLOTTE pipeline end-to-end, choose your path:

### Option 1: Scale Up to Full Experiments
→ **[EXPERIMENTATION.md](EXPERIMENTATION.md)** - Learn how to run full-scale experiments with:
- Larger datasets (10k-100k parallel sentences)
- Full training (5k-20k NMT steps)
- Complete evaluation and analysis
- Production-quality models

### Option 2: Prepare Your Own Data
→ **[DATA_PREPARATION.md](DATA_PREPARATION.md)** - Learn how to:
- Obtain parallel corpora for your language pairs
- Create train/validation/test splits
- Format data for CharLOTTE
- Determine if you have enough data

### Option 3: Understand Configuration Options
→ **[CONFIGURATION.md](CONFIGURATION.md)** - Reference guide for:
- SC model parameters
- Tokenizer settings
- NMT model architecture
- Hyperparameter tuning

### Option 4: Deep Dive into Monitoring
→ **[MONITORING.md](MONITORING.md)** - Learn to:
- Monitor training with TensorBoard
- Interpret training metrics
- Evaluate model quality
- Debug training issues

---

## Appendix: SC-Only Quick Test (Optional)

If you want to test just the SC model training component quickly (without running the full pipeline), you can run a standalone SC training test:

**Time**: ~5-7 minutes on GPU, ~10-15 minutes on CPU
**Purpose**: Verify FastAlign and SC model training work correctly

```bash
cd $SCIMT_DIR/charlotte-test
bash ../Pipeline/train_SC_venv.sh test-sc-es-pt.cfg ../venv_sound ../venv_copper
```

**Expected output**: SC model trained with BLEU > 60 on Spanish→Portuguese cognate prediction.

**Note**: This test only trains the SC model and does NOT demonstrate the complete CharLOTTE methodology. We recommend running the Full Pipeline Test above instead to see the complete end-to-end workflow.

---

**Congratulations!** You've successfully run the complete CharLOTTE pipeline end-to-end. You're now ready to tackle full-scale low-resource NMT experiments with SC augmentation.

**[← Back to README](../README.md)** | **[Full Experiments →](EXPERIMENTATION.md)** | **[Prepare Your Data →](DATA_PREPARATION.md)**
