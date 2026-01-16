# CharLOTTE Project Context

## Project Overview

**CharLOTTE** (Character-Level Orthographic Transfer for Token Embeddings) is a research system for low-resource Neural Machine Translation (NMT) that leverages character correspondences between related languages.

**Core Concept:** Learn systematic sound/orthographic correspondences (like Spanish j→Portuguese lh in "ojo/olho", "ajo/alho") to transform high-resource language data to look more like low-resource related languages, improving NMT vocabulary overlap.

## Repository Structure

```
code/
├── NMT/                    # Neural Machine Translation pipeline (PyTorch Lightning)
│   ├── train.py           # Main NMT training script
│   ├── LightningBART.py   # BART-based transformer models
│   ├── parallel_datasets.py
│   ├── char_tokenizers.py
│   └── configs/CONFIGS/   # YAML configs per language pair
│
├── SC/                     # Sound Correspondence models (NEW - under development)
│   └── [To be implemented - PyTorch Lightning GRU seq2seq]
│
├── Pipeline/               # SC model training orchestration (LEGACY - being replaced)
│   ├── train_SC.sh        # 500+ line bash script for SC training
│   ├── pred_SC.sh         # SC model inference
│   ├── cfg/SC/            # Old .cfg config files
│   └── *.py               # Python helper scripts
│
├── word_alignments/        # FastAlign interface and cognate extraction
│   ├── prepare_for_fastalign.py
│   ├── make_word_alignments_no_grouping.py
│   └── make_cognate_list.py
│
├── CopperMT/              # LEGACY: Fairseq-based RNN models (being replaced)
│   └── CopperMT/pipeline/neural_translation/
│
└── CognateProductionRNN/  # Older simple RNN implementation
```

## Key Workflows

### NMT Training (Production Pipeline)
```bash
python NMT/train.py --config NMT/configs/CONFIGS/mfe-en/baseline.yaml --mode TRAIN
```

### SC Model Training (LEGACY - bash-based)
```bash
bash Pipeline/train_SC.sh Pipeline/cfg/SC/es-an.cfg
```

### SC Model Prediction (LEGACY)
```bash
bash Pipeline/pred_SC.sh Pipeline/cfg/SC/es-an.cfg
```

## Architecture Patterns

### NMT Models (Current Standard)
- **Framework:** PyTorch Lightning
- **Model:** BART-based transformers (encoder-decoder)
- **Tokenization:** SentencePiece (subword) OR character-level
- **Data:** CSV files with `[src_lang, tgt_lang, src_path, tgt_path]`
- **Config:** YAML files in `NMT/configs/CONFIGS/`
- **Training:** Multi-GPU DDP, early stopping, checkpoint selection

### SC Models (Legacy - CopperMT)
- **Framework:** Fairseq (old)
- **Model:** BiGRU encoder + GRU decoder with Luong/Bahdanau attention
- **Tokenization:** Pure character-level (words → "h e l l o")
- **Training:** Bash orchestration, 287 hyperparameter configs
- **Purpose:** Character-to-character transformation for cognates

### SC Models (New - In Development)
- **Framework:** PyTorch Lightning (following NMT patterns)
- **Model:** Same GRU architecture, modernized implementation
- **Goal:** Replace bash scripts with clean Python pipeline

## Important Conventions

### Character-Level Tokenization
```python
# For SC models - preserves orthographic patterns
"hello" → ['h', 'e', 'l', 'l', 'o']
# Tokenizer: NMT/char_tokenizers.py (CharacterTokenizer class)
```

### Data Format (CSV)
```csv
src_lang,tgt_lang,src_path,tgt_path
es,en,/path/to/train.es,/path/to/train.en
fr,en,/path/to/train.fr,/path/to/train.en
```

### Config Files
- **NMT:** YAML format in `NMT/configs/CONFIGS/<lang-pair>/`
- **SC (legacy):** Bash configs in `Pipeline/cfg/SC/`
- **SC (new):** YAML format in `SC/configs/` (to be created)

### Language Codes
Standard 2-3 letter codes: `en`, `es`, `fr`, `mfe`, `djk`, `an`, `as`, `hi`, etc.

## Key Dependencies

### FastAlign (External Tool)
- **Location:** `../fast_align/build/` (outside this repo)
- **Purpose:** Word alignment for cognate extraction
- **Interface:** Scripts in `word_alignments/`
- **Keep:** Yes, works well for cognate detection

### Python Libraries
- PyTorch + PyTorch Lightning
- Transformers (HuggingFace)
- SentencePiece
- sacrebleu (BLEU scoring)
- Fairseq (legacy - CopperMT only)

### Tokenizers
- NLTK, spaCy, IndicNLP, CamelTools (language-specific)
- Used in: `word_alignments/prepare_for_fastalign.py`

## Common Tasks

### Add a New NMT Language Pair
1. Create data CSV files (train/val/test)
2. Create config: `NMT/configs/CONFIGS/<src>-<tgt>/baseline.yaml`
3. Train tokenizer: `Pipeline/train_srctgt_tokenizer.sh`
4. Train model: `python NMT/train.py --config ...`

### Train an SC Model (Legacy)
1. Create config: `Pipeline/cfg/SC/<src>-<tgt>.cfg`
2. Extract cognates: `bash Pipeline/train_SC.sh <config>`
3. Apply to data: `bash Pipeline/pred_SC.sh <config>`

### Extract Cognates from Parallel Data
1. Prepare data: `word_alignments/prepare_for_fastalign.py`
2. Run FastAlign (forward, reverse, symmetrized)
3. Extract pairs: `word_alignments/make_word_alignments_no_grouping.py`
4. Filter by edit distance: `word_alignments/make_cognate_list.py`

## Current State & Migration

### ✅ Production-Ready
- NMT pipeline (PyTorch Lightning BART models)
- Character tokenizers
- Evaluation metrics (BLEU, chrF)
- FastAlign cognate extraction

### 🚧 Under Migration
- SC model pipeline: CopperMT (Fairseq) → PyTorch Lightning
- Config format: Bash .cfg → YAML
- Orchestration: Bash scripts → Python CLI

### 📦 Legacy (Do Not Extend)
- `Pipeline/train_SC.sh` - Being replaced
- `Pipeline/pred_SC.sh` - Being replaced
- `CopperMT/` - Fairseq dependency
- `NMT/hr_CopperMT.py` - Complex 869-line inference wrapper

## File Naming Conventions

### Model Outputs
- SC-transformed data: `original.SC_{MODEL_ID}_{SRC}2{TGT}.ext`
- Example: `train.fr.SC_FR-MFE-v1_fr2mfe.txt`

### Checkpoints
- NMT: `lightning_logs/version_X/checkpoints/epoch=Y-step=Z.ckpt`
- SC (legacy): `{COPPERMT_DATA_DIR}/{SRC}_{TGT}_{TYPE}-{ID}_S-{SEED}/workspace/...`

### Vocabularies
- Character: `{save_dir}/vocab.{lang}.csv`
- SentencePiece: `{TOK_TRAIN_DATA_DIR}/{src}_{tgt}/model.{model|vocab}`

## Performance Notes

### GPU Usage
- NMT training: Typically 4 GPUs with DDP
- SC training: Single GPU sufficient (smaller models)
- SLURM: Use `srun` for multi-GPU, see `NMT/sbatch/`

### Typical Hyperparameters

**NMT (BART):**
```yaml
encoder_layers: 6
encoder_attention_heads: 8
d_model: 512
learning_rate: 5e-4
batch_size: 128
max_steps: 350000
```

**SC (GRU - Legacy):**
```
encoder: BiGRU, 2 layers, hidden=512, embed=256
decoder: GRU, 2 layers, hidden=512, embed=256
attention: Bahdanau
batch_size: 128
learning_rate: 1e-3
```

## Metrics

### NMT Evaluation
- BLEU (sacrebleu)
- chrF
- COMET (optional)

### SC Evaluation
- Character-level BLEU (on cognate test set)
- chrF
- Edit distance to target

## Common Gotchas

1. **Character tokenizer vocab building:** Must provide data paths on first run
2. **FastAlign requires compilation:** Check `../fast_align/build/` exists
3. **SLURM notices:** Ignore system messages about Julia, archive, etc.
4. **Cognate threshold:** 0.5 normalized edit distance is typical
5. **SC model confidence:** Use `log_p_thresh` to filter low-confidence predictions
6. **Data contamination:** Ensure no overlap between train/val/test at word level

## Code Style

- Type hints preferred
- PyTorch Lightning for training loops
- YAML for configs
- CSV for data manifests
- Avoid bash when possible (Python CLI preferred)

## Testing

Currently minimal automated testing. When adding tests:
- Unit tests for models: `SC/tests/test_model.py`
- Integration tests for pipelines
- Validation: Compare with CopperMT baseline (±2% BLEU tolerance)

## Resources

- README: `/home/hatch5o6/Cognate/code/README.md` (comprehensive pipeline docs)
- NMT configs: `NMT/configs/CONFIGS/`
- SLURM scripts: `NMT/sbatch/`
- Hyperparameter search: `Pipeline/rnn_hyperparams/` (287 configs)

## Migration Plan

See `/home/hatch5o6/.claude/plans/cosmic-stargazing-orbit.md` for detailed SC pipeline reimplementation plan.

**TL;DR:** Replace CopperMT (Fairseq + bash) with PyTorch Lightning GRU models following NMT patterns.

---

## Matching Fairseq's SacreBLEU Scoring in evaluate.py

### Problem
Need to ensure `Pipeline/evaluate.py` scores BLEU exactly the same way as fairseq-generate does with sacrebleu to get consistent results.

### How Fairseq Uses SacreBLEU

#### Key Files in Fairseq

**1. Scorer Implementation**
`/home/hatch5o6/miniconda3/envs/copper/lib/python3.8/site-packages/fairseq/scoring/bleu.py:30-68`

The `SacrebleuScorer` class (registered as "sacrebleu"):
- Uses `EvaluationTokenizer` with three configurable parameters:
  - `--sacrebleu-tokenizer` (default: '13a')
  - `--sacrebleu-lowercase` (default: False)
  - `--sacrebleu-char-level` (default: False)
- Pre-tokenizes strings using the tokenizer
- Calls `sacrebleu.corpus_bleu(self.pred, [self.ref], tokenize="none")`
- Note: `tokenize="none"` because tokenization is already done

**2. Tokenizer Implementation**
`/home/hatch5o6/miniconda3/envs/copper/lib/python3.8/site-packages/fairseq/scoring/tokenizer.py:9-66`

The `EvaluationTokenizer` class:
- Uses sacrebleu's built-in TOKENIZERS
- Default is '13a' (standard WMT tokenization)
- Applies tokenization, then optionally punctuation removal, character tokenization, and lowercasing

**3. Main Generation Script**
`/home/hatch5o6/miniconda3/envs/copper/lib/python3.8/site-packages/fairseq_cli/generate.py:172`

- Line 172: `scorer = scoring.build_scorer(args, tgt_dict)`
- Lines 334-337: Adds reference and hypothesis strings to scorer
- Line 368: Prints final BLEU score

**4. Your Shell Script**
`CopperMT/CopperMT/pipeline/neural_translation/checkpoint_select_best.sh:44-51`

Uses `--scoring sacrebleu` with **default parameters** (13a tokenizer, no lowercasing, no character-level).

### Solution: Matching Fairseq Exactly

#### Final Working Code for calc_bleu

```python
import sacrebleu
from sacrebleu.tokenizers import Tokenizer13a

def calc_bleu(
    hyp, # list
    refs # list of lists
):
    for ref in refs:
        if len(hyp) != len(ref):
            error = f"len hyp ({len(hyp)}) != len ref ({len(ref)})"
            error += "HYP:\n" + "\n".join(hyp[:3])
            error += "\nREF:\n" + "\n".join(ref[:3])
            raise ValueError(error)

    # Match fairseq exactly: use 13a tokenizer, then pass tokenize="none"
    tokenizer = Tokenizer13a()
    tokenized_hyp = [tokenizer(h) for h in hyp]
    tokenized_refs = []
    for ref_set in refs:
        tokenized_refs.append([tokenizer(r) for r in ref_set])

    # Use function-based API (sacrebleu 1.5.1)
    score = sacrebleu.corpus_bleu(tokenized_hyp, tokenized_refs, tokenize="none")
    print("BLEU STUFF")
    print(score)
    return score.score
```

#### Key Changes Made

1. **Import**: Changed from `BLEU` class to function-based API
   - `import sacrebleu`
   - `from sacrebleu.tokenizers import Tokenizer13a`

2. **Tokenization**: Pre-tokenize all strings with 13a tokenizer
   - Initialize: `tokenizer = Tokenizer13a()`
   - Apply to hyps: `tokenized_hyp = [tokenizer(h) for h in hyp]`
   - Apply to refs: `tokenized_refs.append([tokenizer(r) for r in ref_set])`

3. **Scoring**: Use function-based API with `tokenize="none"`
   - `sacrebleu.corpus_bleu(tokenized_hyp, tokenized_refs, tokenize="none")`

#### Why This Matches Fairseq

1. ✅ Uses '13a' tokenizer (same as fairseq default)
2. ✅ Pre-tokenizes all strings before scoring
3. ✅ Passes `tokenize="none"` to sacrebleu (since already tokenized)
4. ✅ No lowercasing (fairseq default)
5. ✅ No character-level evaluation (fairseq default)

#### chrF Unchanged

The `calc_chrF` function was kept as-is because:
- chrF is a character-level metric
- Should work on raw, untokenized strings
- Word tokenization would change what chrF measures
- Original implementation was already correct

### Environment Notes

- **copper environment**: Used for fairseq commands, sacrebleu 1.5.1
- **sound environment**: Previously used for evaluate.py, sacrebleu 1.5.1
- Both have same sacrebleu version, but needed to use function-based API for compatibility
- Now running evaluate.py in copper environment for consistency

### Default SacreBLEU Configuration

When using `--scoring sacrebleu` in fairseq-generate:
- Tokenizer: '13a' (standard WMT tokenization)
- Lowercase: False
- Character-level: False
- tokenize parameter in corpus_bleu: "none" (tokenization handled beforehand)

### CRITICAL: Reference File Mismatch Issue

**Problem:** Even after fixing tokenization, BLEU scores still don't match fairseq.

**Root Cause:** Fairseq scores against **different references** than raw input files.

#### What Fairseq Actually Scores

Fairseq-generate outputs multiple lines per sentence:
```
S-0    [source sentence]
T-0    [reference - from preprocessed data-bin]
H-0    [score] [hypothesis - tokenized]
D-0    [score] [hypothesis - detokenized]
P-0    [positional scores]
```

On line 335 of fairseq_cli/generate.py:
```python
scorer.add_string(target_str, detok_hypo_str)
```

Fairseq scores: **T- lines (from data-bin)** vs **D- lines (detokenized hypothesis)**

Note: H- and D- lines are identical (asserted in NMT/hr_CopperMT.py:503)

#### What You Were Scoring

```python
REF: .../inputs/split_data/es_an/0/fine_tune_es_an.an  # Raw input file
HYP: .../generate-valid.hyp.txt  # Extracted H- lines
```

#### Why They Differ

The preprocessed data-bin references (T- lines) can differ from raw files because:
1. **Preprocessing filters** (length limits, empty lines removed)
2. **Normalization** (unicode, whitespace handling)
3. **Reordering** of sentences
4. **Dropped sentences** that failed validation

#### The Fix

**Extract both T- and H- lines from fairseq output:**

```python
# In NMT/hr_CopperMT.py or evaluation script
refs = []  # T- lines from fairseq output
hyps = []  # H- lines from fairseq output

for S, T, H, D, P in tqdm(data):
    T = T.split("\t")[-1].strip()
    H = H.split("\t")[-1].strip()

    # Remove spaces if needed
    ref = "".join(T.split()) if not RETURN_SPACED else T
    hyp = "".join(H.split()) if not RETURN_SPACED else H

    refs.append(ref)
    hyps.append(hyp)

# Score exactly what fairseq scored
bleu = calc_bleu(hyps, [refs])
```

**Key Insight:** Always extract references from the T- lines in fairseq's output, not from the original input files. This ensures you're scoring against the exact same references that fairseq used from the preprocessed data-bin.

#### Verification Checklist

To match fairseq BLEU scores exactly:
- ✅ Use 13a tokenizer with `tokenize="none"`
- ✅ Extract **both** T- lines (refs) and H-/D- lines (hyps) from fairseq output
- ✅ Apply same spacing/joining as fairseq (check RETURN_SPACED flag)
- ✅ Ensure sentence order matches (use IDs from S-/T-/H-/D- prefixes)
- ✅ Run in same environment (copper) with same sacrebleu version
