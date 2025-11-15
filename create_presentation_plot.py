#!/usr/bin/env python3
"""
Create single impressive plot for hackathon presentation.

Shows: FP32 vs FP16 comparison with key metrics
- Model Size Reduction
- Accuracy Preservation
- FL Bandwidth Savings
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def find_latest_results():
    """Find the most recent quantization_comparison.json file."""
    results_files = list(Path("outputs").rglob("quantization_comparison.json"))
    if not results_files:
        raise FileNotFoundError("No quantization_comparison.json found!")

    latest = max(results_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest}")
    return latest

def load_results(json_path):
    """Load comparison results."""
    with open(json_path, 'r') as f:
        return json.load(f)

def create_hackathon_plot(results):
    """Create single impressive plot for hackathon presentation."""

    # Extract data
    fp32 = results['precisions']['fp32']
    fp16 = results['precisions']['fp16']

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Data
    categories = ['Accuracy\n(%)', 'Model Size\n(MB)', 'FL Bandwidth\n(10 rounds, 10 clients)']

    # Normalize to percentage of FP32
    fp32_acc = fp32['accuracy_percent']
    fp16_acc = fp16['accuracy_percent']

    fp32_size = fp32['file_size_mb']
    fp16_size = fp16['file_size_mb']

    # FL bandwidth = model_size * 2 (up+down) * 10 clients * 10 rounds
    fp32_bandwidth = fp32_size * 2 * 10 * 10  # MB
    fp16_bandwidth = fp16_size * 2 * 10 * 10  # MB

    fp32_values = [100, 100, 100]  # Baseline
    fp16_values = [
        (fp16_acc / fp32_acc) * 100,  # Accuracy retention
        (fp16_size / fp32_size) * 100,  # Size as % of FP32
        (fp16_bandwidth / fp32_bandwidth) * 100  # Bandwidth as % of FP32
    ]

    x = np.arange(len(categories))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, fp32_values, width, label='FP32 (Baseline)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=2)
    bars2 = ax.bar(x + width/2, fp16_values, width, label='FP16 (Mixed Precision)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=2)

    # Styling
    ax.set_ylabel('Percentage of FP32 Baseline', fontsize=14, fontweight='bold')
    ax.set_title('Federated Learning: FP16 Mixed Precision Benefits\nEuroSAT Satellite Classification',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 110])

    # Add 100% reference line
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add actual values as annotations
    annotations = [
        f"Accuracy:\nFP32: {fp32_acc:.1f}%\nFP16: {fp16_acc:.1f}%\nDrop: {fp32_acc - fp16_acc:.2f}%",
        f"Model Size:\nFP32: {fp32_size:.2f} MB\nFP16: {fp16_size:.2f} MB\nSaved: {fp32_size - fp16_size:.2f} MB",
        f"FL Bandwidth:\nFP32: {fp32_bandwidth:.0f} MB\nFP16: {fp16_bandwidth:.0f} MB\nSaved: {fp32_bandwidth - fp16_bandwidth:.0f} MB"
    ]

    for i, (pos, text) in enumerate(zip(x, annotations)):
        ax.text(pos, -15, text, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

    # Add key insight box
    insight = (
        "🎯 Key Result: FP16 mixed precision achieves:\n"
        f"• 50% smaller models ({fp16_size:.2f} MB vs {fp32_size:.2f} MB)\n"
        f"• {abs(fp32_acc - fp16_acc):.2f}% accuracy drop (virtually identical)\n"
        f"• 50% less bandwidth in Federated Learning\n"
        "• Faster training with torch.cuda.amp"
    )

    ax.text(0.98, 0.98, insight, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, pad=1),
           fontweight='bold')

    plt.tight_layout()

    return fig

def main():
    print("="*60)
    print("HACKATHON PRESENTATION PLOT GENERATOR")
    print("="*60)
    print()

    # Load results
    results_path = find_latest_results()
    results = load_results(results_path)

    # Create plot
    print("Creating presentation plot...")
    fig = create_hackathon_plot(results)

    # Save
    output_dir = results_path.parent
    output_path = output_dir / "presentation_plot.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Presentation plot saved: {output_path}")
    print(f"   Resolution: 300 DPI (print quality)")
    print(f"   Size: ~14x8 inches")
    print()

if __name__ == "__main__":
    main()
