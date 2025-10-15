#!/usr/bin/env python3
"""
Plot SC model training and validation loss curves from fairseq log output.

Usage:
    python plot_sc_training_loss.py <log_file> [output.png]

Example:
    python plot_sc_training_loss.py logs/quickstart_phase1_sc_training_20251015_113320.log sc_loss_curves.png
"""

import sys
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

def parse_fairseq_log(log_file):
    """
    Parse fairseq training log to extract loss and perplexity metrics.

    Returns:
        dict with keys:
            - train_epochs: list of epoch numbers
            - train_loss: list of average training loss per epoch
            - train_ppl: list of average training perplexity per epoch
            - val_epochs: list of epoch numbers for validation
            - val_loss: list of validation loss
            - val_ppl: list of validation perplexity
            - train_steps: list of step numbers
            - train_loss_step: list of training loss at each logged step
    """
    train_epochs = []
    train_loss = []
    train_ppl = []
    val_epochs = []
    val_loss = []
    val_ppl = []
    train_steps = []
    train_loss_step = []

    # Patterns to match
    # Example: 2025-10-15 11:33:44 | INFO | train | epoch 001 | loss 3.663 | ... | ppl 12.67 | ...
    epoch_pattern = re.compile(
        r'INFO \| train \| epoch (\d+) \| loss ([\d.]+) \| .* \| ppl ([\d.]+)'
    )

    # Example: 2025-10-15 11:33:44 | INFO | valid | epoch 001 | valid on 'valid' subset | loss 2.001 | ... | ppl 4 | ...
    valid_pattern = re.compile(
        r'INFO \| valid \| epoch (\d+) \| .* \| loss ([\d.]+) \| .* \| ppl ([\d.]+)'
    )

    # Example: 2025-10-15 11:33:32 | INFO | train_inner | epoch 001:    100 / 487 loss=4.655, ... num_updates=100, ...
    step_pattern = re.compile(
        r'INFO \| train_inner \| epoch \d+:\s+\d+ / \d+ loss=([\d.]+),.* num_updates=(\d+),'
    )

    with open(log_file, 'r') as f:
        for line in f:
            # Check for end-of-epoch training metrics
            match = epoch_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                ppl = float(match.group(3))
                train_epochs.append(epoch)
                train_loss.append(loss)
                train_ppl.append(ppl)

            # Check for validation metrics
            match = valid_pattern.search(line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                ppl = float(match.group(3))
                val_epochs.append(epoch)
                val_loss.append(loss)
                val_ppl.append(ppl)

            # Check for step-level training metrics
            match = step_pattern.search(line)
            if match:
                loss = float(match.group(1))
                step = int(match.group(2))
                train_steps.append(step)
                train_loss_step.append(loss)

    return {
        'train_epochs': train_epochs,
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'val_epochs': val_epochs,
        'val_loss': val_loss,
        'val_ppl': val_ppl,
        'train_steps': train_steps,
        'train_loss_step': train_loss_step,
    }

def plot_loss_curves(log_file, output_path=None):
    """
    Plot training and validation loss curves from fairseq SC model log.

    Args:
        log_file: Path to fairseq training log file
        output_path: Path to save plot (optional, defaults to same dir as log)
    """
    # Parse the log file
    data = parse_fairseq_log(log_file)

    if not data['train_epochs']:
        print(f"Warning: No training metrics found in {log_file}")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss by step (within-epoch resolution)
    if data['train_steps']:
        ax1.plot(data['train_steps'], data['train_loss_step'],
                alpha=0.6, linewidth=1, label='Training Loss (step)')

    # Add epoch-level averages if available
    if data['train_epochs']:
        # Map epochs to approximate step numbers (use max step per epoch)
        epoch_steps = []
        for epoch in data['train_epochs']:
            # Find the max step number for this epoch
            max_step = max([s for s, _ in zip(data['train_steps'], data['train_loss_step'])
                           if s <= epoch * max(data['train_steps']) / max(data['train_epochs'])],
                          default=epoch * len(data['train_steps']) // len(data['train_epochs']))
            epoch_steps.append(max_step)

        ax1.plot(epoch_steps, data['train_loss'],
                'o-', linewidth=2, markersize=6, label='Training Loss (epoch avg)')

    # Add validation points
    if data['val_epochs']:
        val_steps = [e * max(data['train_steps']) / max(data['train_epochs'])
                    for e in data['val_epochs']]
        ax1.plot(val_steps, data['val_loss'],
                's-', linewidth=2, markersize=8, label='Validation Loss')

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('SC Model Training and Validation Loss by Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss and Perplexity by epoch
    ax2_twin = ax2.twinx()

    if data['train_epochs']:
        line1 = ax2.plot(data['train_epochs'], data['train_loss'],
                'o-', linewidth=2, color='tab:blue', label='Training Loss')
        line2 = ax2_twin.plot(data['train_epochs'], data['train_ppl'],
                's--', linewidth=2, color='tab:orange', label='Training Perplexity')

    if data['val_epochs']:
        line3 = ax2.plot(data['val_epochs'], data['val_loss'],
                's-', linewidth=2, markersize=8, color='tab:green', label='Validation Loss')
        line4 = ax2_twin.plot(data['val_epochs'], data['val_ppl'],
                'D--', linewidth=2, markersize=6, color='tab:red', label='Validation Perplexity')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2_twin.set_ylabel('Perplexity', color='tab:orange')
    ax2_twin.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_title('SC Model Training and Validation Loss by Epoch')
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    if data['val_epochs']:
        lines += line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper right')

    plt.tight_layout()

    # Save the plot
    if output_path is None:
        log_path = Path(log_file)
        output_path = log_path.parent / f'{log_path.stem}_loss_curves.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ SC model loss curves saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("SC Model Training Summary")
    print("="*50)

    if data['train_epochs']:
        print(f"Total epochs:     {max(data['train_epochs'])}")
        print(f"Total steps:      {max(data['train_steps']) if data['train_steps'] else 'N/A'}")
        print(f"\nInitial training loss: {data['train_loss'][0]:.4f}")
        print(f"Final training loss:   {data['train_loss'][-1]:.4f}")
        print(f"Reduction:             {data['train_loss'][0] - data['train_loss'][-1]:.4f} ({100*(1-data['train_loss'][-1]/data['train_loss'][0]):.1f}%)")

        print(f"\nInitial perplexity:    {data['train_ppl'][0]:.2f}")
        print(f"Final perplexity:      {data['train_ppl'][-1]:.2f}")

    if data['val_epochs']:
        best_val_idx = data['val_loss'].index(min(data['val_loss']))
        print(f"\nBest validation loss:  {data['val_loss'][best_val_idx]:.4f} (epoch {data['val_epochs'][best_val_idx]})")
        print(f"Best val perplexity:   {data['val_ppl'][best_val_idx]:.2f}")
        print(f"Final validation loss: {data['val_loss'][-1]:.4f}")
        print(f"Final val perplexity:  {data['val_ppl'][-1]:.2f}")

    print("="*50)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    log_file = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    plot_loss_curves(log_file, output_path)
