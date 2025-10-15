# CharLOTTE Quick Start

**Purpose:** Run your first end-to-end CharLOTTE experiment - from SC model training through actual Portuguese→English translation.

This guide demonstrates the complete CharLOTTE workflow with a small Spanish→Portuguese→English example, taking you from character correspondence learning all the way to generating and evaluating translations.

**Prerequisites**: Complete [SETUP.md](SETUP.md) installation and verify all components are working (all three virtual environments: venv_sound, venv_copper, venv_nmt).

---

## What You'll Accomplish

Run the complete 6-phase CharLOTTE pipeline end-to-end:
1. **Prepare data** - Download FLORES-200 dataset (automated)
2. **Train SC model** - Learn Spanish→Portuguese character correspondences
3. **Apply SC model** - Transform Spanish data to augment Portuguese training data
4. **Train tokenizer** - Build multilingual vocabulary with SC-augmented data
5. **Train NMT model** - Train Portuguese→English translator with augmentation
6. **Evaluate & translate** - Generate actual English translations and evaluate quality

**Time**: ~30-45 minutes (mostly automated)
**Hardware**: GPU recommended (8-10x faster), CPU supported

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
- Download FLORES-200 data if needed (800 train, 100 val, 100 test sentences)
- Run all 6 phases sequentially
- Skip already-completed phases
- Save detailed logs for each phase
- Generate actual English translations from Portuguese

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
Translating test set (100 sentences)...
BLEU: 24.5, chrF: 52.3
✓ Phase 5 Complete: Translations generated
```

### Step 4: Verify Results

Check the generated translations:
```bash
# View sample translations (Portuguese → English)
head -5 nmt_models/pt-en/translations/test.pt-en.hyp

# Compare with reference translations
head -5 data/raw/test.pt-en.en
```

**Expected BLEU Score**: 20-30 (this is a quick test on very limited data)

All outputs are saved in:
- `sc_models_es_pt/` - Trained SC model
- `spm_models/SC_es2pt-pt_en/` - Trained tokenizer
- `nmt_models/pt-en/` - Trained NMT model and translations
- `quickstart_phase*.log` - Detailed logs for each phase

### Step 5: Understand Your Results

**SC Model Quality** (Phase 1):
- BLEU > 60: ✅ Excellent - strong character correspondences learned
- BLEU 40-60: ✅ Good - captures most systematic patterns
- BLEU < 40: ⚠️ Check if languages are actually related

**NMT Translation Quality** (Phase 5):
- BLEU 20-30: ✅ Expected for this small-scale test (800 training sentences)
- BLEU 15-20: ✅ Reasonable - SC augmentation is helping
- BLEU < 15: ⚠️ Check logs for training issues

**Key Takeaway**: Even with very limited data (800 training sentences), CharLOTTE's SC augmentation enables functional translation. Full experiments with 10k-50k sentences achieve much higher quality

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
