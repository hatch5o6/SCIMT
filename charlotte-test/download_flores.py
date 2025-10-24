#!/usr/bin/env python3
"""
Download FLORES-200 dataset for Spanish, Portuguese, and English
Creates train/val/test splits for SC model training and NMT training
"""

from datasets import load_dataset
import os

# Create output directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/csv", exist_ok=True)

# Load FLORES+ devtest and dev splits to get ~2000 sentences per language
print("Loading FLORES+ dataset (devtest + dev splits)...")
devtest = load_dataset("openlanguagedata/flores_plus", split='devtest')
dev = load_dataset("openlanguagedata/flores_plus", split='dev')

# Combine both splits
from itertools import chain
dataset = list(chain(devtest, dev))

print(f"Total dataset size: {len(dataset)} rows (devtest: {len(devtest)}, dev: {len(dev)})")

# Filter for Spanish (spa), Portuguese (por), and English (eng)
# ISO 639-3 codes: spa = Spanish, por = Portuguese, eng = English
print("Filtering for Spanish, Portuguese, and English...")
flores_es = [item for item in dataset if item['iso_639_3'] == 'spa']
flores_pt = [item for item in dataset if item['iso_639_3'] == 'por']
flores_en = [item for item in dataset if item['iso_639_3'] == 'eng']

print(f"Loaded {len(flores_es)} Spanish sentences")
print(f"Loaded {len(flores_pt)} Portuguese sentences")
print(f"Loaded {len(flores_en)} English sentences")

# Extract sentences
es_sentences = [item['text'] for item in flores_es]
pt_sentences = [item['text'] for item in flores_pt]
en_sentences = [item['text'] for item in flores_en]

# Create splits (1600 train, 200 val, 200 test - doubled from original)
# Note: FLORES+ devtest has ~1000 sentences, so we'll use max available
train_size = 1600
val_size = 200
test_size = 200

print(f"\nCreating splits: {train_size} train, {val_size} val, {test_size} test")

# Write SC training data (es-pt)
with open("data/raw/train.es", "w") as f:
    f.write("\n".join(es_sentences[:train_size]) + "\n")

with open("data/raw/train.pt", "w") as f:
    f.write("\n".join(pt_sentences[:train_size]) + "\n")

# Write validation data (es-pt)
with open("data/raw/val.es", "w") as f:
    f.write("\n".join(es_sentences[train_size:train_size+val_size]) + "\n")

with open("data/raw/val.pt", "w") as f:
    f.write("\n".join(pt_sentences[train_size:train_size+val_size]) + "\n")

# Write test data (es-pt)
with open("data/raw/test.es", "w") as f:
    f.write("\n".join(es_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

with open("data/raw/test.pt", "w") as f:
    f.write("\n".join(pt_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

# Write NMT training data (pt-en for low-resource, es-en for high-resource)
with open("data/raw/train.pt-en.pt", "w") as f:
    f.write("\n".join(pt_sentences[:train_size]) + "\n")

with open("data/raw/train.pt-en.en", "w") as f:
    f.write("\n".join(en_sentences[:train_size]) + "\n")

with open("data/raw/train.es-en.es", "w") as f:
    f.write("\n".join(es_sentences[:train_size]) + "\n")

with open("data/raw/train.es-en.en", "w") as f:
    f.write("\n".join(en_sentences[:train_size]) + "\n")

# Write NMT validation data
with open("data/raw/val.pt-en.pt", "w") as f:
    f.write("\n".join(pt_sentences[train_size:train_size+val_size]) + "\n")

with open("data/raw/val.pt-en.en", "w") as f:
    f.write("\n".join(en_sentences[train_size:train_size+val_size]) + "\n")

with open("data/raw/val.es-en.es", "w") as f:
    f.write("\n".join(es_sentences[train_size:train_size+val_size]) + "\n")

with open("data/raw/val.es-en.en", "w") as f:
    f.write("\n".join(en_sentences[train_size:train_size+val_size]) + "\n")

# Write NMT test data
with open("data/raw/test.pt-en.pt", "w") as f:
    f.write("\n".join(pt_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

with open("data/raw/test.pt-en.en", "w") as f:
    f.write("\n".join(en_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

with open("data/raw/test.es-en.es", "w") as f:
    f.write("\n".join(es_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

with open("data/raw/test.es-en.en", "w") as f:
    f.write("\n".join(en_sentences[train_size+val_size:train_size+val_size+test_size]) + "\n")

print(f"\nCreated data files:")
print(f"  SC training: train.es, train.pt ({train_size} sentences)")
print(f"  SC validation: val.es, val.pt ({val_size} sentences)")
print(f"  SC test: test.es, test.pt ({test_size} sentences)")
print(f"  NMT training: train.pt-en.*, train.es-en.* ({train_size} sentences each)")
print(f"  NMT validation: val.pt-en.*, val.es-en.* ({val_size} sentences each)")
print(f"  NMT test: test.pt-en.*, test.es-en.* ({test_size} sentences each)")

# Create CSV config files with absolute paths
base_path = os.path.abspath("data/raw")

# SC Training CSV files
with open("data/csv/train.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"es,pt,{base_path}/train.es,{base_path}/train.pt\n")

with open("data/csv/val.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"es,pt,{base_path}/val.es,{base_path}/val.pt\n")

with open("data/csv/test.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"es,pt,{base_path}/test.es,{base_path}/test.pt\n")

# NMT Training CSV files
with open("data/csv/nmt_train.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"pt,en,{base_path}/train.pt-en.pt,{base_path}/train.pt-en.en\n")
    f.write(f"es,en,{base_path}/train.es-en.es,{base_path}/train.es-en.en\n")

with open("data/csv/nmt_val.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"pt,en,{base_path}/val.pt-en.pt,{base_path}/val.pt-en.en\n")
    f.write(f"es,en,{base_path}/val.es-en.es,{base_path}/val.es-en.en\n")

with open("data/csv/nmt_test.csv", "w") as f:
    f.write("src_lang,tgt_lang,src_path,tgt_path\n")
    f.write(f"pt,en,{base_path}/test.pt-en.pt,{base_path}/test.pt-en.en\n")

print(f"\nCreated CSV config files with absolute paths:")
print(f"  Base path: {base_path}")
print(f"  SC CSVs: train.csv, val.csv, test.csv")
print(f"  NMT CSVs: nmt_train.csv, nmt_val.csv, nmt_test.csv")
print("\nData preparation complete!")
