# CharLOTTE Training Monitoring and Evaluation Guide

**Purpose:** Learn how to monitor training progress, interpret metrics, and evaluate model quality.

**Prerequisites**: Review [EXPERIMENTATION.md](EXPERIMENTATION.md) to understand the training workflow.

---

## Table of Contents

1. [Monitoring SC Model Training](#monitoring-sc-model-training)
2. [Monitoring NMT Training](#monitoring-nmt-training)
3. [Evaluating SC Models](#evaluating-sc-models)
4. [Evaluating NMT Models](#evaluating-nmt-models)
5. [Comparing Models](#comparing-models)
6. [Advanced Evaluation](#advanced-evaluation)

---

## Monitoring SC Model Training

### Terminal Output

When training an SC model with `train_SC.sh` or `train_SC_venv.sh`, you'll see three phases:

#### Phase 1: Cognate Extraction
```
Extracting cognates using FastAlign...
Found 5,432 cognate pairs with threshold 0.5
Split: train=4,345, val=543, test=544
```

**What to check**:
- **Number of cognates**: Should be at least 500 pairs
  - < 500 pairs: Increase `COGNATE_THRESH` or verify languages are related
  - 500-2,000 pairs: Minimal but workable
  - 2,000-10,000 pairs: Good
  - 10,000+ pairs: Excellent

#### Phase 2: SC Model Training (RNN)
```
| epoch 001 | loss 2.345 | ppl 10.43 | accuracy 0.423
| epoch 002 | loss 1.876 | ppl 6.53 | accuracy 0.512
...
| epoch 015 | loss 0.963 | ppl 2.62 | accuracy 0.687  (best checkpoint)
| epoch 020 | loss 0.869 | ppl 2.38 | accuracy 0.712
done training in 245 seconds
```

**What to watch for**:
- **Loss decreasing**: Should drop from ~2-4 to < 1.5
- **PPL (perplexity) decreasing**: Should drop below 3
- **Accuracy increasing**: Should reach 60-80%
- **Best checkpoint**: Saved based on validation loss

#### Phase 3: SC Model Evaluation
```
| Eval on test set | test_loss 1.084 | test_ppl 2.96
BLEU (character-level): 68.32
chrF: 61.52
```

**What good scores look like**:
- BLEU > 60: ✅ Excellent
- BLEU 40-60: ✅ Good
- BLEU 30-40: ⚠️ Acceptable but could be better
- BLEU < 30: ❌ Poor - check if languages are actually related

### SC Training Log Files

Logs are saved to the SC model output directory:
```
$BASE_DIR/models/sc_models/es_pt_RNN-0_S-1000/
├── train.log
├── checkpoints/
│   ├── checkpoint1.pt
│   ├── checkpoint_best.pt
│   └── checkpoint_last.pt
└── test_results.txt
```

**Useful commands**:
```bash
# Watch training progress
tail -f $BASE_DIR/models/sc_models/es_pt_*/train.log

# Check final test scores
grep "BLEU\|chrF" $BASE_DIR/models/sc_models/es_pt_*/test_results.txt
```

### Common SC Training Issues

**Issue**: Too few cognates extracted
**Solution**:
- Increase `COGNATE_THRESH` (try 0.6 or 0.7)
- Verify languages are related
- Check that parallel data quality is good

**Issue**: Loss not decreasing
**Solution**:
- Check hyperparameter settings
- Verify cognate quality (inspect extracted pairs)
- Try different `RNN_HYPERPARAMS_ID`

**Issue**: Low test BLEU (< 30)
**Solution**:
- Languages may not be closely related enough
- Cognate pairs may be noisy
- Try SMT model instead of RNN

---

## Monitoring NMT Training

NMT training produces comprehensive logs using PyTorch Lightning.

### Output Directory Structure

```
$BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/
├── checkpoints/
│   ├── epoch=0-step=1000.ckpt
│   ├── epoch=1-step=2000.ckpt
│   └── best_model.ckpt
├── logs/
│   └── version_0/
│       ├── events.out.tfevents.XXX  # TensorBoard logs
│       ├── metrics.csv               # Training metrics
│       └── hparams.yaml             # Hyperparameters used
├── predictions/
│   ├── test_predictions.txt
│   └── metrics.json
└── config.yaml                      # Config copy
```

### Real-Time Monitoring with CSV Metrics

**View metrics while training**:
```bash
# Watch training progress
tail -f $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/logs/version_0/metrics.csv

# Extract validation metrics
grep "val_loss" $BASE_DIR/.../logs/version_0/metrics.csv | tail -20
```

**Example metrics.csv output**:
```csv
epoch,step,train_loss,val_loss,val_bleu,learning_rate
0,1000,3.456,3.234,0.0,0.0002
0,2000,3.123,2.987,0.0,0.0002
1,3000,2.876,2.765,12.3,0.0002
1,4000,2.654,2.543,14.6,0.0002
```

### Monitoring with TensorBoard

**Launch TensorBoard**:
```bash
# Activate virtual environment
source $SCIMT_DIR/venv_sound/bin/activate

# Launch TensorBoard
tensorboard --logdir $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/logs

# Output: TensorBoard 2.X.X at http://localhost:6006/
```

**Access TensorBoard**:
1. Open browser to `http://localhost:6006/`
2. Navigate tabs:
   - **Scalars**: Loss curves, BLEU scores, learning rate
   - **Text**: Sample translations (if logged)
   - **Time Series**: Training speed and GPU utilization

### Key Metrics to Monitor

| Metric | What It Shows | Healthy Behavior |
|--------|---------------|------------------|
| `train_loss` | Training loss per batch | Steadily decreasing (3.5 → 1.5 → 0.8) |
| `val_loss` | Validation loss | Should decrease with train_loss, may plateau |
| `val_bleu` | BLEU on validation set | Increasing (0 → 20-40+) |
| `learning_rate` | Current learning rate | May decay over training (if using scheduler) |
| `epoch` | Current epoch | Increments over time |

### Terminal Output During Training

**Example training output**:
```
Epoch 0: 100%|██████████| 1000/1000 [12:34<00:00, 1.32it/s, loss=2.876]
Validation: 100%|██████████| 50/50 [00:45<00:00, 1.11it/s]
Epoch 0, Step 1000: train_loss=2.876, val_loss=2.543, val_bleu=14.6

Epoch 1: 100%|██████████| 1000/1000 [12:31<00:00, 1.33it/s, loss=2.654]
Validation: 100%|██████████| 50/50 [00:44<00:00, 1.14it/s]
Epoch 1, Step 2000: train_loss=2.654, val_loss=2.312, val_bleu=17.2
```

**Progress bar breakdown**:
- `Epoch X`: Current epoch number
- `100%`: Progress through epoch
- `██████████`: Visual progress bar
- `1000/1000`: Steps completed / total steps
- `[12:34<00:00, 1.32it/s]`: Time elapsed, time remaining, iterations/sec
- `loss=2.876`: Current training loss

### Monitoring GPU Utilization

**Check GPU usage**:
```bash
# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -i 1
```

**Expected GPU utilization**:
- **GPU Memory**: 50-95% (depends on batch size and model size)
- **GPU Utilization**: 80-100% during training steps
- **Temperature**: < 85°C (normal operating range)

**If GPU utilization is low** (< 50%):
- Data loading bottleneck → Increase dataloader workers
- Batch size too small → Increase `train_batch_size`
- CPU preprocessing slow → Check tokenization speed

### When to Stop Training

**Early Stopping (Automatic)**:
CharLOTTE uses early stopping with patience (default: 10 epochs). Training stops when validation loss doesn't improve for `early_stop` epochs.

**Manual Stopping**:
Press `Ctrl+C`. The best checkpoint so far will be saved.

**Signs of Good Training**:
- Validation loss decreasing steadily
- Validation BLEU increasing to 15-40+ (depends on data size)
- Training and validation loss both decreasing (no overfitting)
- Training completes in expected time (hours to days)

**Resuming Training**:
```yaml
# In NMT config
from_pretrained: /path/to/checkpoint.ckpt
```

---

## Evaluating SC Models

SC models are automatically evaluated during training. Results are printed and logged.

### SC Evaluation Metrics

| Metric | Description | Good Score |
|--------|-------------|------------|
| **Character-level BLEU** | BLEU on character sequences | > 60 |
| **chrF** | Character n-gram F-score | > 60 |
| **Accuracy** | Exact match accuracy | > 30% |

### Example SC Evaluation Output

```
Testing SC model on test set...
Test BLEU (character-level): 68.32
Test chrF: 61.52
Test Accuracy: 34.2%

Sample predictions:
  Source (es):  hijo
  Target (pt):  filho
  Predicted:    filho  ✓

  Source (es):  hierro
  Target (pt):  ferro
  Predicted:    ferro  ✓

  Source (es):  noche
  Target (pt):  noite
  Predicted:    noite  ✓
```

### Interpreting SC Model Scores

**BLEU scores**:
- **< 30**: SC model struggling - check language relatedness
- **30-50**: Reasonable performance - captures some patterns
- **> 50**: Good performance - learns most correspondences
- **> 70**: Excellent performance - very accurate transformations

**chrF typically 5-15 points higher than BLEU**: This is normal for character-level evaluation.

### Manual Testing of SC Models

Test SC models on custom word lists using CopperMT's prediction scripts. Consult CopperMT documentation for details.

---

## Evaluating NMT Models

After training, run evaluation on the test set.

### Running NMT Evaluation

```bash
cd $SCIMT_DIR/NMT
python train.py -c $BASE_DIR/configs/nmt/pt-en.PRETRAIN.yaml -m TEST
```

**What happens**:
1. Loads best checkpoint from training
2. Runs inference on test set
3. Computes BLEU and chrF scores
4. Saves predictions and metrics

### Output Files

```
$BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/predictions/
├── test_predictions.txt          # Model outputs
├── test_references.txt            # Ground truth
├── test_sources.txt               # Input sources
└── metrics.json                   # BLEU, chrF, etc.
```

**Example metrics.json**:
```json
{
  "test_bleu": 28.4,
  "test_chrf": 54.2,
  "test_loss": 1.876,
  "num_samples": 2000
}
```

### Understanding NMT Metrics

**BLEU (Bilingual Evaluation Understudy)**:
- Measures n-gram overlap with reference translations
- Range: 0-100 (higher is better)
- Standard metric for machine translation

**chrF (Character n-gram F-score)**:
- Character-level n-gram F-score
- Range: 0-100 (higher is better)
- More robust for morphologically rich languages

### Typical Score Ranges

| Data Size | Expected BLEU | Expected chrF | Quality |
|-----------|---------------|---------------|---------|
| 5k (low-resource only) | 5-15 | 25-40 | Poor |
| 5k + 100k augmented (with SC) | 15-30 | 40-60 | Usable |
| 50k+ sentences | 25-40 | 50-70 | Good |
| 500k+ sentences | 35-50 | 65-80 | Excellent |

**Note**: Scores vary by:
- Language pair difficulty (English-French easier than English-Japanese)
- Domain (news easier than literary texts)
- Language relatedness (related pairs benefit more from SC)

### Inspecting Predictions

**View sample predictions**:
```bash
# Show first 20 predictions
head -20 $BASE_DIR/models/nmt_models/pt-en/PRETRAIN_TRIAL_s=1000/predictions/test_predictions.txt

# Compare side-by-side
paste test_sources.txt test_predictions.txt test_references.txt | head -10
```

**Example output**:
```
Source:      Este é um exemplo em português.
Predicted:   This is an example in Portuguese.
Reference:   This is an example in Portuguese.
Status:      ✓ Perfect match

Source:      O tempo está bom hoje.
Predicted:   The weather is good today.
Reference:   The weather is fine today.
Status:      ~ Acceptable (synonym)
```

### Qualitative Analysis

Beyond automatic metrics, perform manual evaluation:

**1. Fluency**: Are outputs grammatically correct?
- Good: "I want to study quantum physics."
- Bad: "I want study physics quantum."

**2. Adequacy**: Does translation preserve meaning?
- Good: "The weather is fine today." (preserves meaning)
- Bad: "Today is a day." (loses meaning)

**3. Terminology**: Domain-specific terms correct?
- Check technical terms, proper nouns, numbers, dates

**4. Common Errors**:
- **Hallucination**: Model generates content not in source
- **Omission**: Model skips parts of source
- **Repetition**: Model repeats phrases
- **Mistranslation**: Incorrect word choice

**Systematic error analysis**:
```bash
# Random sample for manual review
paste test_sources.txt test_predictions.txt test_references.txt | shuf | head -50
```

---

## Comparing Models

### Baseline vs. SC-Augmented Comparison

To measure SC augmentation impact, train two models:

**1. Baseline (no SC)**:
```yaml
# NMT config
sc_model_id: null  # No SC augmentation
```

**2. SC-Augmented**:
```yaml
# NMT config
sc_model_id: es2pt-RNN-0-RNN-0  # With SC augmentation
```

**Compare test scores**:
```
Baseline BLEU:        18.3
SC-Augmented BLEU:    28.6
Improvement:          +10.3 BLEU points (56% relative improvement)
```

### Statistical Significance Testing

For research, run multiple random seeds:

**Train with different seeds**:
```yaml
# Config 1
seed: 1000

# Config 2
seed: 2000

# Config 3
seed: 3000
```

**Report**: "BLEU 28.6 ± 1.2 (mean ± std over 3 seeds)"

### Using Custom Test Sets

**Evaluate on domain-specific data**:

1. **Prepare custom test CSV**:
```csv
src_lang,tgt_lang,src_path,tgt_path
pt,en,/path/to/domain_test.src,/path/to/domain_test.tgt
```

2. **Update config**:
```yaml
test_data: /path/to/domain_test.csv
```

3. **Run evaluation**:
```bash
python train.py -c config.yaml -m TEST
```

---

## Advanced Evaluation

### Exporting Results for Publication

**Generate comparison table**:
```bash
# Extract all metrics
cat metrics.json | python -m json.tool

# Create CSV
echo "Model,BLEU,chrF" > results.csv
echo "Baseline,18.3,42.5" >> results.csv
echo "SC-Augmented,28.6,56.3" >> results.csv
```

### Human Evaluation

For research papers:
- Random sample of 100-200 sentences
- 2-3 annotators rate (1-5 scale for fluency/adequacy)
- Report inter-annotator agreement (Kappa or Pearson)

### Per-Sentence Analysis

**Compute per-sentence BLEU** (requires custom script):
```python
from sacrebleu import sentence_bleu

with open('test_predictions.txt') as pred, open('test_references.txt') as ref:
    for p, r in zip(pred, ref):
        score = sentence_bleu(p.strip(), [r.strip()]).score
        if score < 10:  # Flag low-scoring sentences
            print(f"Low BLEU: {score:.1f} | Pred: {p.strip()}")
```

### Error Category Analysis

Manually categorize errors by type:
- Lexical errors (wrong word choice)
- Syntactic errors (grammar issues)
- Semantic errors (meaning lost/changed)
- Stylistic errors (awkward phrasing)

Track error frequency to identify model weaknesses.

---

## Troubleshooting

For detailed troubleshooting of training and evaluation issues, see:

- **[TROUBLESHOOTING.md - Training Issues](TROUBLESHOOTING.md#training-issues)**
- **[TROUBLESHOOTING.md - Evaluation Issues](TROUBLESHOOTING.md#evaluation-issues)**

Common issues covered:
- Loss not decreasing
- Validation loss diverging
- CUDA out of memory
- Training too slow
- Poor BLEU scores
- SC augmentation not helping

---

## Summary

**Key takeaways**:
- Monitor training in real-time with CSV metrics or TensorBoard
- SC models should achieve BLEU > 60 for good quality
- NMT models typically achieve BLEU 15-40 depending on data size
- SC augmentation provides 5-15 BLEU point improvements for related languages
- Manual inspection of predictions is crucial for understanding model behavior

---

**[← Back to README](../README.md)** | **[Configuration →](CONFIGURATION.md)** | **[Troubleshooting →](TROUBLESHOOTING.md)**
