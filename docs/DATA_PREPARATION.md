# CharLOTTE Data Preparation Guide

**Purpose:** Learn how to obtain, format, and prepare parallel datasets for CharLOTTE experiments.

**Prerequisites**: Review [SETUP.md](SETUP.md) to ensure CharLOTTE is installed correctly.

---

## Table of Contents

1. [Dataset Requirements](#dataset-requirements)
2. [Obtaining Training Data](#obtaining-training-data)
3. [Creating Train/Val/Test Splits](#creating-trainvaltest-splits)
4. [Data Format Requirements](#data-format-requirements)
5. [Creating CSV Metadata Files](#creating-csv-metadata-files)
6. [Recommended Directory Structure](#recommended-directory-structure)
7. [Data Quality Checklist](#data-quality-checklist)

---

## Dataset Requirements

### Dataset Size Guidelines

The amount of parallel data directly impacts model quality. Here's what to expect with different data sizes:

#### Low-Resource Language Pair

| Size | Training Feasibility | Notes |
|------|---------------------|-------|
| < 2,000 pairs | ❌ Not recommended | Model barely learns; mostly memorization |
| 2,000-5,000 pairs | ⚠️ Minimal | Proof-of-concept only; poor quality |
| 5,000-10,000 pairs | ✅ Viable | Usable for research; SC augmentation helps significantly |
| 10,000-30,000 pairs | ✅ Good | Practical quality; SC augmentation provides solid boost |
| 30,000-50,000 pairs | ✅ Strong | Good quality; diminishing returns from SC augmentation |
| 50,000+ pairs | ✅ Excellent | High quality; SC augmentation still beneficial but smaller gains |

**For expected BLEU scores by data size**, see [MONITORING.md - Typical Score Ranges](MONITORING.md#typical-score-ranges).

#### High-Resource Language Pair (for SC augmentation)

| Size | SC Model Quality | Augmentation Impact | Notes |
|------|-----------------|---------------------|-------|
| < 50,000 pairs | Poor | Minimal | Not enough data to learn robust character correspondences |
| 50,000-100,000 pairs | Fair | Moderate | SC model learns some patterns; moderate augmentation benefit |
| 100,000-500,000 pairs | Good | Strong | **Recommended minimum** - SC model captures most systematic correspondences |
| 500,000+ pairs | Excellent | Maximum | Ideal scenario - SC model learns comprehensive character mappings |

#### Validation and Test Sets

- **Validation set**: 500-1,000 pairs per language pair (minimum 500 for reliable early stopping)
- **Test set**: 1,000-2,000 pairs per language pair (minimum 1,000 for stable BLEU scores)
- **Important**: Ensure **no overlap** between train/val/test splits

### Practical Examples

**Scenario 1: Very Low-Resource**
- Low-resource: 3,000 Portuguese-English pairs
- High-resource: 200,000 Spanish-English pairs
- **Result**: Without SC augmentation, BLEU ~8. With SC augmentation, BLEU ~15-18. Model barely usable but SC provides 2x improvement.

**Scenario 2: Standard Low-Resource**
- Low-resource: 10,000 Portuguese-English pairs
- High-resource: 300,000 Spanish-English pairs
- **Result**: Without SC augmentation, BLEU ~18. With SC augmentation, BLEU ~28. Practical quality suitable for research or assistance applications.

**Scenario 3: Medium-Resource**
- Low-resource: 40,000 Portuguese-English pairs
- High-resource: 500,000 Spanish-English pairs
- **Result**: Without SC augmentation, BLEU ~25. With SC augmentation, BLEU ~33. High-quality translations with SC providing a solid 8-point boost.

---

## Obtaining Training Data

### Public Parallel Corpora

#### OPUS (Recommended)
**Website**: https://opus.nlpl.eu/

The largest collection of freely available parallel corpora:
- **Content**: OpenSubtitles, Europarl, Wikipedia, Bible translations, TED talks, etc.
- **Coverage**: 100+ languages
- **Formats**: Plain text, TMX, Moses format
- **Size**: Ranges from thousands to millions of sentence pairs

**Example: Downloading Portuguese-English from OPUS**
```bash
# Navigate to OPUS website
# Search for language pair: pt-en
# Select corpus (e.g., OpenSubtitles, Tatoeba)
# Download in TMX or plain text format

# If TMX format, convert to plain text:
# (Use tmx2txt or similar tool)
```

#### Other Sources

**Tatoeba** (https://tatoeba.org/)
- Sentence-level translations
- Good for low-resource languages
- Community-contributed
- High quality but smaller size

**JW300** (Jehovah's Witness translations)
- Available for many low-resource languages
- Religious domain
- ~300k sentences per language pair

**CCMatrix** (Common Crawl mined)
- Mined parallel sentences from web
- Large scale (millions of pairs)
- Quality varies (may need filtering)

**Bible translations**
- Available for 100+ languages
- Religious domain
- ~30k verse pairs

### Language Pair Selection

#### Identifying Related Languages

CharLOTTE works best when languages share systematic character correspondences. Your language pair is likely suitable if:

- **Same language family**:
  - Romance: Spanish/Portuguese, French/Italian, Romanian/Italian
  - Germanic: English/German, Dutch/German, Swedish/Norwegian
  - Slavic: Russian/Polish, Czech/Slovak, Serbian/Croatian
  - Indo-Aryan: Hindi/Urdu, Bengali/Assamese

- **Significant borrowing/contact**:
  - English-French (Norman conquest)
  - Hindi-Urdu (shared vocabulary)
  - Turkish-Arabic (loanwords)

- **Test**: Can you find 20+ word pairs with similar spellings?
  - Spanish-Portuguese: *hijo/filho*, *noche/noite*, *madre/mãe*
  - English-German: *father/Vater*, *mother/Mutter*, *house/Haus*

#### When SC Augmentation Helps Most

✅ **Recommended scenarios**:
- Related languages from same family
- High-resource language has 100k+ parallel sentences
- Low-resource language has < 50k parallel sentences

❌ **Not recommended scenarios**:
- Unrelated languages (e.g., English-Japanese, Arabic-Chinese)
- Different writing systems (Latin vs. Cyrillic vs. logographic)
- Already have 100k+ low-resource sentences (diminishing returns)

---

## Creating Train/Val/Test Splits

Once you have parallel data, split it into train/validation/test sets:

### Python Script for Splitting

```python
import random

def split_parallel_data(src_file, tgt_file, output_prefix, train_ratio=0.8, val_ratio=0.1):
    """
    Split parallel data into train/val/test sets.

    Args:
        src_file: Path to source language file
        tgt_file: Path to target language file
        output_prefix: Prefix for output files (e.g., 'pt-en')
        train_ratio: Proportion for training (default: 0.8)
        val_ratio: Proportion for validation (default: 0.1)
        # test_ratio is implicit: 1 - train_ratio - val_ratio
    """
    # Read data
    with open(src_file, encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    with open(tgt_file, encoding='utf-8') as f:
        tgt_lines = [line.strip() for line in f]

    assert len(src_lines) == len(tgt_lines), \
        f"Source and target must have same length: {len(src_lines)} vs {len(tgt_lines)}"

    # Shuffle together (with fixed seed for reproducibility)
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

        with open(src_out, 'w', encoding='utf-8') as f:
            f.write('\n'.join([s for s, t in split_pairs]) + '\n')
        with open(tgt_out, 'w', encoding='utf-8') as f:
            f.write('\n'.join([t for s, t in split_pairs]) + '\n')

        print(f"{split_name}: {len(split_pairs)} pairs")
        print(f"  Written to: {src_out}, {tgt_out}")

# Example usage:
if __name__ == "__main__":
    split_parallel_data('pt-en.pt', 'pt-en.en', 'portuguese-english')
    # Creates: portuguese-english.train.src, portuguese-english.train.tgt,
    #          portuguese-english.val.src, portuguese-english.val.tgt,
    #          portuguese-english.test.src, portuguese-english.test.tgt
```

### Split Ratios

**Standard split** (for datasets > 10k pairs):
- Train: 80%
- Validation: 10%
- Test: 10%

**For smaller datasets** (< 10k pairs):
- Train: 80%
- Validation: 10%
- Test: 10%
- (Keep absolute validation/test sizes at least 500 pairs each)

**Important notes**:
- Always use fixed random seed for reproducibility
- Ensure no sentence appears in multiple splits
- Verify splits have similar distribution (check sentence length statistics)

---

## Data Format Requirements

### Parallel Text Files

Parallel text files must be **sentence-aligned** with **one sentence per line**:

**Example: train.src (Portuguese)**
```
Este é um exemplo em português.
O tempo está bom hoje.
Gostaria de aprender sobre tradução automática.
```

**Example: train.tgt (English)**
```
This is an example in Portuguese.
The weather is good today.
I would like to learn about machine translation.
```

**Key requirements**:
- ✅ One sentence per line
- ✅ Line N in source corresponds to line N in target
- ✅ UTF-8 encoding
- ✅ No empty lines (or ensure source and target have empty lines at same positions)
- ✅ Consistent tokenization (spaces between words)

**Common issues to avoid**:
- ❌ Different number of lines in source and target
- ❌ Multi-line sentences
- ❌ Inconsistent encoding (mixing UTF-8, Latin-1, etc.)
- ❌ Extra whitespace or tabs

### Data Quality Check Script

```python
def check_parallel_data(src_file, tgt_file):
    """Verify parallel data quality."""
    with open(src_file, encoding='utf-8') as f:
        src_lines = f.readlines()
    with open(tgt_file, encoding='utf-8') as f:
        tgt_lines = f.readlines()

    print(f"Source lines: {len(src_lines)}")
    print(f"Target lines: {len(tgt_lines)}")

    if len(src_lines) != len(tgt_lines):
        print(f"ERROR: Line count mismatch!")
        return False

    # Check for empty lines
    empty_src = [i for i, line in enumerate(src_lines) if not line.strip()]
    empty_tgt = [i for i, line in enumerate(tgt_lines) if not line.strip()]

    if empty_src:
        print(f"WARNING: {len(empty_src)} empty source lines at positions: {empty_src[:5]}...")
    if empty_tgt:
        print(f"WARNING: {len(empty_tgt)} empty target lines at positions: {empty_tgt[:5]}...")

    # Check length statistics
    src_lengths = [len(line.split()) for line in src_lines]
    tgt_lengths = [len(line.split()) for line in tgt_lines]

    print(f"Source avg length: {sum(src_lengths)/len(src_lengths):.1f} words")
    print(f"Target avg length: {sum(tgt_lengths)/len(tgt_lengths):.1f} words")

    return True

# Example usage
check_parallel_data('train.src', 'train.tgt')
```

---

## Creating CSV Metadata Files

CharLOTTE uses CSV files to specify parallel data locations.

### CSV Format

**Header** (required):
```csv
src_lang,tgt_lang,src_path,tgt_path
```

**Example: data/csv/train.no_overlap_v1.csv**
```csv
src_lang,tgt_lang,src_path,tgt_path
pt,en,/absolute/path/to/data/raw/low-resource/train.src,/absolute/path/to/data/raw/low-resource/train.tgt
es,en,/absolute/path/to/data/raw/high-resource/train.src,/absolute/path/to/data/raw/high-resource/train.tgt
```

**Important requirements**:
- ✅ Use **absolute paths**, not relative paths
- ✅ Paths must not contain `~` (tilde) - expand to full path
- ✅ Use forward slashes `/` even on Windows
- ✅ Training CSV must be named `train.no_overlap_v1.csv`
- ✅ Validation CSV must be named `val.no_overlap_v1.csv`
- ✅ Test CSV must be named `test.csv`

### CSV Naming Convention

The standard naming is `train.no_overlap_v1.csv` where:
- **no_overlap**: Train/val/test splits have zero sentence overlap (required for valid evaluation)
- **v1**: Version 1 of your data splits (increment if you create new splits: v2, v3, etc.)

**Can you use different names?** Yes, but:
- ✅ **Recommended**: Use these standard names for easier documentation following
- ⚠️ **Custom names**: Requires updating all config file references (SC, tokenizer, NMT)
  - Example: If you use `train.mydata.csv`, you must update:
    - `PARALLEL_TRAIN=` in SC config files
    - `TRAIN_PARALLEL=` in tokenizer config files
    - `train_data:` in NMT YAML files

**Why it matters**: Standard names make it easier to follow examples and reduce config errors. The CharLOTTE codebase expects these names in many examples.

### Python Script for Creating CSVs

```python
import csv
import os

# Configuration
BASE_DIR = "/absolute/path/to/your-project"
DATA_DIR = f"{BASE_DIR}/data"
RAW_DIR = f"{DATA_DIR}/raw"
CSV_DIR = f"{DATA_DIR}/csv"

# Language pair configuration
LOW_RESOURCE_SRC = "pt"  # Portuguese
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

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['src_lang', 'tgt_lang', 'src_path', 'tgt_path'])

        # Low-resource data (all splits)
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

---

## Recommended Directory Structure

Organize your project with this structure:

```
your-project/
├── data/
│   ├── raw/                          # Raw parallel text files
│   │   ├── low-resource/
│   │   │   ├── train.src            # Portuguese training data
│   │   │   ├── train.tgt            # English training data
│   │   │   ├── val.src
│   │   │   ├── val.tgt
│   │   │   ├── test.src
│   │   │   └── test.tgt
│   │   └── high-resource/
│   │       ├── train.src            # Spanish training data
│   │       ├── train.tgt            # English training data
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

---

## Data Quality Checklist

Before starting experiments, verify:

### File Format
- [ ] Source and target files have same number of lines
- [ ] Files are UTF-8 encoded
- [ ] No empty lines (or matching empty lines in both files)
- [ ] One sentence per line
- [ ] Consistent tokenization

### Data Splits
- [ ] Train/val/test splits have no overlap
- [ ] Validation set has at least 500 pairs
- [ ] Test set has at least 1,000 pairs
- [ ] Splits created with fixed random seed

### CSV Files
- [ ] All paths are absolute (no `~` or relative paths)
- [ ] All files referenced in CSV actually exist
- [ ] CSV files named correctly (`train.no_overlap_v1.csv`, etc.)
- [ ] Language codes are correct (ISO 639-1: en, es, pt, etc.)

### Language Pair Suitability
- [ ] Languages are related or have contact
- [ ] Can identify 20+ cognate word pairs manually
- [ ] High-resource language has 100k+ pairs (for SC augmentation)
- [ ] Low-resource language has 5k+ pairs (minimum for viable NMT)

### Quick Validation Commands

```bash
# Check line counts match
wc -l data/raw/low-resource/train.src data/raw/low-resource/train.tgt

# Check encoding
file -i data/raw/low-resource/train.src

# Check for empty lines
grep -c "^$" data/raw/low-resource/train.src

# Verify CSV paths exist
while IFS=, read -r src_lang tgt_lang src_path tgt_path; do
    [ -f "$src_path" ] || echo "Missing: $src_path"
    [ -f "$tgt_path" ] || echo "Missing: $tgt_path"
done < data/csv/train.no_overlap_v1.csv
```

---

## Next Steps

Once your data is prepared and passes the quality checklist:

1. **Run experiments** → [EXPERIMENTATION.md](EXPERIMENTATION.md)
2. **Understand configurations** → [CONFIGURATION.md](CONFIGURATION.md)
3. **Monitor training** → [MONITORING.md](MONITORING.md)

---

**[← Back to README](../README.md)** | **[Quick Start →](QUICKSTART.md)** | **[Run Experiments →](EXPERIMENTATION.md)**
