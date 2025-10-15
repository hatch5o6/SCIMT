#!/usr/bin/env python3
"""
Plot training and validation loss curves from PyTorch Lightning metrics CSV.

Usage:
    python plot_training_loss.py <metrics.csv> [output.png]

Example:
    python plot_training_loss.py nmt_models/pt-en_TRIAL_s=1000/logs/lightning_logs/version_0/metrics.csv loss_curves.png
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_curves(metrics_csv, output_path=None):
    """
    Plot training and validation loss curves from Lightning metrics CSV.

    Args:
        metrics_csv: Path to metrics.csv file
        output_path: Path to save plot (optional, defaults to same dir as CSV)
    """
    # Read the CSV
    df = pd.read_csv(metrics_csv)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss by step
    train_steps = df[df['train_loss_step'].notna()]
    val_steps = df[df['val_loss_step'].notna()]

    if not train_steps.empty:
        ax1.plot(train_steps['step'], train_steps['train_loss_step'],
                label='Training Loss', alpha=0.6, linewidth=1)

    if not val_steps.empty:
        ax1.plot(val_steps['step'], val_steps['val_loss_step'],
                label='Validation Loss', alpha=0.8, linewidth=2)

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss by Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Loss by epoch (aggregated)
    train_epochs = df[df['train_loss_epoch'].notna()]
    val_epochs = df[df['val_loss_epoch'].notna()]

    if not train_epochs.empty:
        ax2.plot(train_epochs['epoch'], train_epochs['train_loss_epoch'],
                marker='o', label='Training Loss (avg)', linewidth=2)

    if not val_epochs.empty:
        ax2.plot(val_epochs['epoch'], val_epochs['val_loss_epoch'],
                marker='s', label='Validation Loss', linewidth=2)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss by Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add learning rate if available
    lr_data = df[df['lr-AdamW'].notna()]
    if not lr_data.empty:
        ax3 = ax1.twinx()
        ax3.plot(lr_data['step'], lr_data['lr-AdamW'],
                'g--', alpha=0.5, linewidth=1, label='Learning Rate')
        ax3.set_ylabel('Learning Rate', color='g')
        ax3.tick_params(axis='y', labelcolor='g')
        ax3.legend(loc='upper right')

    plt.tight_layout()

    # Save the plot
    if output_path is None:
        output_path = Path(metrics_csv).parent / 'loss_curves.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Loss curves saved to: {output_path}")

    # Print summary statistics
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)

    if not train_epochs.empty:
        final_train = train_epochs.iloc[-1]['train_loss_epoch']
        initial_train = train_epochs.iloc[0]['train_loss_epoch'] if len(train_epochs) > 0 else None
        print(f"Initial training loss: {initial_train:.4f}" if initial_train else "Initial training loss: N/A")
        print(f"Final training loss:   {final_train:.4f}")

    if not val_epochs.empty:
        final_val = val_epochs.iloc[-1]['val_loss_epoch']
        best_val = val_epochs['val_loss_epoch'].min()
        best_epoch = val_epochs.loc[val_epochs['val_loss_epoch'].idxmin(), 'epoch']
        print(f"\nBest validation loss:  {best_val:.4f} (epoch {int(best_epoch)})")
        print(f"Final validation loss: {final_val:.4f}")

    if not lr_data.empty:
        final_lr = lr_data.iloc[-1]['lr-AdamW']
        print(f"\nFinal learning rate:   {final_lr:.2e}")

    total_steps = df['step'].max()
    total_epochs = df['epoch'].max()
    print(f"\nTotal steps:  {int(total_steps)}")
    print(f"Total epochs: {int(total_epochs)}")
    print("="*50)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    metrics_csv = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(metrics_csv).exists():
        print(f"Error: Metrics file not found: {metrics_csv}")
        sys.exit(1)

    plot_loss_curves(metrics_csv, output_path)
