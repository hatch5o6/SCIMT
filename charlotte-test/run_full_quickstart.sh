#!/bin/bash
# Full CharLOTTE Pipeline Quickstart Test
# This script demonstrates the complete end-to-end CharLOTTE workflow:
# 1. Train SC model (Spanish->Portuguese character transformations)
# 2. Apply SC model to transform Spanish data
# 3. Train multilingual tokenizer with SC-augmented data
# 4. Train NMT model (Portuguese->English with SC augmentation)
# 5. Generate translations and evaluate
# 6. BASELINE: Train NMT without SC augmentation for comparison

# Don't exit on error - we want to see all phases even if some fail
# set -e

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PIPELINE_DIR="${SCRIPT_DIR}/../Pipeline"
SC_CONFIG="${SCRIPT_DIR}/test-sc-es-pt.cfg"
TOK_CONFIG="${SCRIPT_DIR}/test-tok-es-pt-en.cfg"
NMT_CONFIG="${SCRIPT_DIR}/test-nmt-pt-en.yaml"
# Baseline configurations (no SC augmentation)
TOK_CONFIG_BASELINE="${SCRIPT_DIR}/test-tok-pt-en-baseline.cfg"
NMT_CONFIG_BASELINE="${SCRIPT_DIR}/test-nmt-pt-en-baseline.yaml"
VENV_SOUND="${SCRIPT_DIR}/../venv_sound"
VENV_COPPER="${SCRIPT_DIR}/../venv_copper"
VENV_NMT="${SCRIPT_DIR}/../venv_nmt"

# Create logs directory and timestamp for log files
LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOGS_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set up main log file and redirect all output to it (and console)
MAIN_LOG="${LOGS_DIR}/quickstart_full_run_${TIMESTAMP}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "=========================================="
echo "CharLOTTE Full Pipeline Quickstart Test"
echo "=========================================="
echo "Log file: ${MAIN_LOG}"
echo ""

# Check if data exists, if not download FLORES
if [ ! -f "${SCRIPT_DIR}/data/raw/train.pt-en.pt" ]; then
    echo "Data files not found. Downloading FLORES dataset..."
    python3 "${SCRIPT_DIR}/download_flores.py"
    echo "✓ FLORES data downloaded successfully"
    echo ""
fi

# Phase 1: Train SC Model
echo "=========================================="
echo "PHASE 1: Training SC Model (es->pt)"
echo "=========================================="

# Check if SC model already exists
SC_MODEL_PATH="${SCRIPT_DIR}/sc_models_es_pt/es_pt_RNN-default_S-1000"
if [ -d "${SC_MODEL_PATH}/workspace/reference_models/bilingual/rnn_es-pt/1000/results" ]; then
    echo "✓ SC model already trained, skipping Phase 1"
    echo "   Model found at: ${SC_MODEL_PATH}"
else
    echo "This trains a character-level RNN to learn Spanish->Portuguese transformations"
    echo "Training on 800 parallel sentences..."
    echo ""

    bash "${PIPELINE_DIR}/train_SC_venv.sh" "${SC_CONFIG}" "${VENV_SOUND}" "${VENV_COPPER}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase1_sc_training_${TIMESTAMP}.log"

    echo ""
    echo "✓ Phase 1 Complete: SC model trained"

    # Generate SC model loss curves if training just completed
    SC_LOG="${LOGS_DIR}/quickstart_phase1_sc_training_${TIMESTAMP}.log"
    if [ -f "${SC_LOG}" ]; then
        echo ""
        echo "Generating SC model loss curves..."
        python "${SCRIPT_DIR}/plot_sc_training_loss.py" "${SC_LOG}" 2>/dev/null

        SC_LOSS_CURVES="${LOGS_DIR}/quickstart_phase1_sc_training_${TIMESTAMP}_loss_curves.png"
        if [ -f "${SC_LOSS_CURVES}" ]; then
            echo "✓ SC model loss curves saved to: ${SC_LOSS_CURVES}"
        fi
    fi
fi
echo ""

# Phase 2: Apply SC Model
echo "=========================================="
echo "PHASE 2: Applying SC Model"
echo "=========================================="

# Check if SC transformations already exist
if [ -f "${SCRIPT_DIR}/data/raw/train.es-en.es.es_pt_RNN-default_S-1000" ]; then
    echo "✓ SC transformations already applied, skipping Phase 2"
    echo "   Transformed file exists: train.es-en.es.es_pt_RNN-default_S-1000"
else
    echo "Transforming Spanish text to Portuguese-like text using trained SC model..."
    echo ""

    # Hardcode the SC model path based on known configuration
    # The model is created as: es_pt_RNN-default_S-1000
    SC_MODEL_PATH="${SCRIPT_DIR}/sc_models_es_pt/es_pt_RNN-default_S-1000"

    echo "SC Model: ${SC_MODEL_PATH}"

    # Apply SC model to Spanish training data
    bash "${PIPELINE_DIR}/pred_SC.sh" "${SC_CONFIG}" "${SCRIPT_DIR}/data/raw/train.es-en.es" "${VENV_SOUND}" "${VENV_COPPER}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase2_sc_apply_${TIMESTAMP}.log"

    echo ""
    echo "✓ Phase 2 Complete: SC transformations applied"
fi
echo ""

# Phase 3: Train Tokenizer
echo "=========================================="
echo "PHASE 3: Training Tokenizer"
echo "=========================================="

# Check if tokenizer already exists
if [ -f "${SCRIPT_DIR}/spm_models/SC_es2pt-pt_en/SC_es2pt-pt_en.model" ]; then
    echo "✓ Tokenizer already trained, skipping Phase 3"
    echo "   Tokenizer found at: spm_models/SC_es2pt-pt_en/SC_es2pt-pt_en.model"
else
    echo "Training multilingual SentencePiece tokenizer on:"
    echo "  - SC-transformed Spanish (es2pt)"
    echo "  - Portuguese (pt)"
    echo "  - English (en)"
    echo ""

    # Activate venv_sound and call train_tokenizer.sh (which doesn't handle venv arg properly)
    source "${VENV_SOUND}/bin/activate"
    bash "${PIPELINE_DIR}/train_tokenizer.sh" "${TOK_CONFIG}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase3_tokenizer_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase 3 Complete: Tokenizer trained"
fi
echo ""

# Check if NMT venv exists
if [ ! -d "${VENV_NMT}" ]; then
    echo "✗ ERROR: NMT virtual environment not found at ${VENV_NMT}"
    echo "✗ The full CharLOTTE pipeline requires venv_nmt for NMT training and translation"
    echo "✗ Please create venv_nmt following the instructions in docs/SETUP.md"
    echo ""
    echo "To create venv_nmt:"
    echo "  cd ${SCRIPT_DIR}/.."
    echo "  python3.10 -m venv venv_nmt"
    echo "  source venv_nmt/bin/activate"
    echo "  pip install -r nmt.requirements.txt"
    echo ""
    exit 1
fi

# Phase 4: Train NMT Model
echo "=========================================="
echo "PHASE 4: Training NMT Model (pt->en)"
echo "=========================================="

# Check if NMT model already trained
NMT_MODEL_DIR="${SCRIPT_DIR}/nmt_models/pt-en_TRIAL_s=1000"
if [ -d "${NMT_MODEL_DIR}/checkpoints" ] && [ -n "$(ls -A ${NMT_MODEL_DIR}/checkpoints/*.ckpt 2>/dev/null)" ]; then
    echo "✓ NMT model already trained, skipping Phase 4"
    echo "   Checkpoints found at: ${NMT_MODEL_DIR}/checkpoints"
else
    echo "Training transformer NMT model with CharLOTTE methodology"
    echo "Training for 500 steps (quick test)..."
    echo ""

    source "${VENV_NMT}/bin/activate"
    python "${PIPELINE_DIR}/../NMT/train.py" -c "${NMT_CONFIG}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase4_nmt_training_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase 4 Complete: NMT model trained"
fi
echo ""

# Phase 5: Evaluate and Generate Translations
echo "=========================================="
echo "PHASE 5: Generating Translations"
echo "=========================================="

# Check if translations already generated
PREDICTIONS_DIR="${NMT_MODEL_DIR}/predictions"
if [ -f "${PREDICTIONS_DIR}/all_scores.json" ]; then
    echo "✓ Translations already generated, skipping Phase 5"
    echo "   Results found at: ${PREDICTIONS_DIR}"
    echo ""
    echo "BLEU scores:"
    grep -A 3 "BEST_BLEU_CHECKPOINT" "${PREDICTIONS_DIR}/all_scores.json" || echo "   (Check ${PREDICTIONS_DIR}/all_scores.json for scores)"
else
    echo "Translating test set and computing BLEU scores..."
    echo ""

    source "${VENV_NMT}/bin/activate"
    python "${PIPELINE_DIR}/../NMT/train.py" -c "${NMT_CONFIG}" -m TEST 2>&1 | tee "${LOGS_DIR}/quickstart_phase5_translation_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase 5 Complete: Translations generated"
fi
echo ""

# Generate loss curves
echo "=========================================="
echo "Generating Loss Curves"
echo "=========================================="

METRICS_CSV="${NMT_MODEL_DIR}/logs/lightning_logs/version_0/metrics.csv"
if [ -f "${METRICS_CSV}" ]; then
    echo "Plotting training and validation loss curves..."
    source "${VENV_NMT}/bin/activate"
    python "${SCRIPT_DIR}/plot_training_loss.py" "${METRICS_CSV}" 2>/dev/null
    deactivate

    LOSS_CURVES="${NMT_MODEL_DIR}/logs/lightning_logs/version_0/loss_curves.png"
    if [ -f "${LOSS_CURVES}" ]; then
        echo "✓ Loss curves saved to: ${LOSS_CURVES}"
    fi
else
    echo "⚠ No metrics.csv found, skipping loss curve generation"
fi
echo ""

# ============================================================================
# BASELINE COMPARISON (No SC Augmentation)
# ============================================================================

echo ""
echo "=========================================="
echo "BASELINE: Training WITHOUT SC Augmentation"
echo "=========================================="
echo "Training a baseline model using ONLY Portuguese-English data"
echo "(No Spanish, no SC transformations)"
echo ""

# Phase B1: Train Baseline Tokenizer
echo "=========================================="
echo "PHASE B1: Training Baseline Tokenizer (pt-en only)"
echo "=========================================="

# Check if baseline tokenizer already exists
if [ -f "${SCRIPT_DIR}/spm_models/pt_en/pt_en.model" ]; then
    echo "✓ Baseline tokenizer already trained, skipping Phase B1"
    echo "   Tokenizer found at: spm_models/pt_en/pt_en.model"
else
    echo "Training SentencePiece tokenizer on Portuguese-English only (NO SC data)..."
    echo ""

    source "${VENV_SOUND}/bin/activate"
    bash "${PIPELINE_DIR}/train_tokenizer.sh" "${TOK_CONFIG_BASELINE}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase_b1_baseline_tokenizer_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase B1 Complete: Baseline tokenizer trained"
fi
echo ""

# Phase B2: Train Baseline NMT Model
echo "=========================================="
echo "PHASE B2: Training Baseline NMT Model (pt->en, NO augmentation)"
echo "=========================================="

# Check if baseline NMT model already trained
BASELINE_NMT_MODEL_DIR="${SCRIPT_DIR}/nmt_models/pt-en_BASELINE_s=1000"
if [ -d "${BASELINE_NMT_MODEL_DIR}/checkpoints" ] && [ -n "$(ls -A ${BASELINE_NMT_MODEL_DIR}/checkpoints/*.ckpt 2>/dev/null)" ]; then
    echo "✓ Baseline NMT model already trained, skipping Phase B2"
    echo "   Checkpoints found at: ${BASELINE_NMT_MODEL_DIR}/checkpoints"
else
    echo "Training baseline NMT model WITHOUT SC augmentation"
    echo "Training on Portuguese-English data only (1600 sentences)..."
    echo ""

    source "${VENV_NMT}/bin/activate"
    python "${PIPELINE_DIR}/../NMT/train.py" -c "${NMT_CONFIG_BASELINE}" 2>&1 | tee "${LOGS_DIR}/quickstart_phase_b2_baseline_nmt_training_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase B2 Complete: Baseline NMT model trained"
fi
echo ""

# Phase B3: Evaluate Baseline Model
echo "=========================================="
echo "PHASE B3: Evaluating Baseline Model"
echo "=========================================="

# Check if baseline translations already generated
BASELINE_PREDICTIONS_DIR="${BASELINE_NMT_MODEL_DIR}/predictions"
if [ -f "${BASELINE_PREDICTIONS_DIR}/all_scores.json" ]; then
    echo "✓ Baseline translations already generated, skipping Phase B3"
    echo "   Results found at: ${BASELINE_PREDICTIONS_DIR}"
    echo ""
    echo "Baseline BLEU scores:"
    grep -A 3 "BEST_BLEU_CHECKPOINT" "${BASELINE_PREDICTIONS_DIR}/all_scores.json" || echo "   (Check ${BASELINE_PREDICTIONS_DIR}/all_scores.json for scores)"
else
    echo "Translating test set with baseline model..."
    echo ""

    source "${VENV_NMT}/bin/activate"
    python "${PIPELINE_DIR}/../NMT/train.py" -c "${NMT_CONFIG_BASELINE}" -m TEST 2>&1 | tee "${LOGS_DIR}/quickstart_phase_b3_baseline_translation_${TIMESTAMP}.log"
    deactivate

    echo ""
    echo "✓ Phase B3 Complete: Baseline translations generated"
fi
echo ""

# Generate baseline loss curves
echo "=========================================="
echo "Generating Baseline Loss Curves"
echo "=========================================="

BASELINE_METRICS_CSV="${BASELINE_NMT_MODEL_DIR}/logs/lightning_logs/version_0/metrics.csv"
if [ -f "${BASELINE_METRICS_CSV}" ]; then
    echo "Plotting baseline training and validation loss curves..."
    source "${VENV_NMT}/bin/activate"
    python "${SCRIPT_DIR}/plot_training_loss.py" "${BASELINE_METRICS_CSV}" 2>/dev/null
    deactivate

    BASELINE_LOSS_CURVES="${BASELINE_NMT_MODEL_DIR}/logs/lightning_logs/version_0/loss_curves.png"
    if [ -f "${BASELINE_LOSS_CURVES}" ]; then
        echo "✓ Baseline loss curves saved to: ${BASELINE_LOSS_CURVES}"
    fi
else
    echo "⚠ No baseline metrics.csv found, skipping loss curve generation"
fi
echo ""

# Summary
echo "=========================================="
echo "QUICKSTART TEST COMPLETE!"
echo "=========================================="
echo ""
echo "SC-AUGMENTED MODEL (CharLOTTE):"
echo "  - SC Model: ${SC_MODEL_PATH}"
echo "  - Tokenizer: ${SCRIPT_DIR}/spm_models/SC_es2pt-pt_en/"
echo "  - NMT Model: ${SCRIPT_DIR}/nmt_models/pt-en_TRIAL_s=1000/"
echo "  - Translations: ${SCRIPT_DIR}/nmt_models/pt-en_TRIAL_s=1000/predictions/"
echo ""
echo "BASELINE MODEL (No SC Augmentation):"
echo "  - Tokenizer: ${SCRIPT_DIR}/spm_models/pt_en_baseline/"
echo "  - NMT Model: ${SCRIPT_DIR}/nmt_models/pt-en_BASELINE_s=1000/"
echo "  - Translations: ${SCRIPT_DIR}/nmt_models/pt-en_BASELINE_s=1000/predictions/"
echo ""
echo "Loss curves:"
# Find SC loss curves (most recent if multiple)
SC_LOSS_CURVES=$(ls -t ${LOGS_DIR}/quickstart_phase1_sc_training_*_loss_curves.png 2>/dev/null | head -1)
if [ -f "${SC_LOSS_CURVES}" ]; then
    echo "  - SC Model: ${SC_LOSS_CURVES}"
fi
if [ -f "${LOSS_CURVES}" ]; then
    echo "  - SC-Augmented NMT: ${LOSS_CURVES}"
fi
if [ -f "${BASELINE_LOSS_CURVES}" ]; then
    echo "  - Baseline NMT: ${BASELINE_LOSS_CURVES}"
fi
echo ""
echo "=========================================="
echo "COMPARISON: SC-Augmented vs Baseline"
echo "=========================================="
echo ""

# Extract and compare BLEU scores
if [ -f "${PREDICTIONS_DIR}/all_scores.json" ] && [ -f "${BASELINE_PREDICTIONS_DIR}/all_scores.json" ]; then
    echo "SC-Augmented Model (with Spanish data via SC transformations):"
    grep -A 2 "BEST_BLEU_CHECKPOINT" "${PREDICTIONS_DIR}/all_scores.json" | grep "BLEU" | head -1

    echo ""
    echo "Baseline Model (Portuguese-English only, NO augmentation):"
    grep -A 2 "BEST_BLEU_CHECKPOINT" "${BASELINE_PREDICTIONS_DIR}/all_scores.json" | grep "BLEU" | head -1

    echo ""
    echo "Dataset sizes:"
    echo "  - SC-Augmented: 1600 pt-en + 1600 es-en (SC-transformed) = 3200 sentence pairs"
    echo "  - Baseline: 1600 pt-en only = 1600 sentence pairs"
    echo ""
    echo "NOTE: With such limited data (1600-3200 sentences), translation quality"
    echo "      will be poor for both models. The comparison demonstrates:"
    echo "      1. Whether SC augmentation helps even with tiny datasets"
    echo "      2. The complete CharLOTTE methodology end-to-end"
    echo ""
    echo "For production-quality translations, use 10k-50k+ sentence pairs"
    echo "(see EXPERIMENTATION.md for full-scale experiments)"
else
    echo "⚠ Could not find scores for comparison"
    echo "   Check: ${PREDICTIONS_DIR}/all_scores.json"
    echo "   Check: ${BASELINE_PREDICTIONS_DIR}/all_scores.json"
fi

echo ""
echo "Log files in ${LOGS_DIR}:"
echo "  - quickstart_full_run_${TIMESTAMP}.log (main log)"
echo "  SC-Augmented Pipeline:"
echo "    - quickstart_phase1_sc_training_${TIMESTAMP}.log"
echo "    - quickstart_phase2_sc_apply_${TIMESTAMP}.log"
echo "    - quickstart_phase3_tokenizer_${TIMESTAMP}.log"
echo "    - quickstart_phase4_nmt_training_${TIMESTAMP}.log"
echo "    - quickstart_phase5_translation_${TIMESTAMP}.log"
echo "  Baseline Pipeline:"
echo "    - quickstart_phase_b1_baseline_tokenizer_${TIMESTAMP}.log"
echo "    - quickstart_phase_b2_baseline_nmt_training_${TIMESTAMP}.log"
echo "    - quickstart_phase_b3_baseline_translation_${TIMESTAMP}.log"
echo ""
echo "You have successfully run the complete CharLOTTE pipeline with baseline comparison!"
echo "See EXPERIMENTATION.md for details on running full-scale experiments."
