"""Create 2x2 grid plot comparing PCA circles for different frequency configurations."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Task1'))

import torch
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ablation import AblationExperiments
from pca_analysis import EmbeddingPCAAnalysis
from utils.common import set_seed, get_device
from dataset import ModularAdditionDataset

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    print("ERROR: TransformerLens not installed")
    sys.exit(1)


def main():
    """Create 2x2 grid of PCA circles for frequency configurations."""
    TASK_DIR = Path(".")
    set_seed(999)
    device = get_device()

    print("="*80)
    print("FREQUENCY COMPARISON VISUALIZATION (PCA CIRCLES)")
    print("="*80)

    P = 113
    FREQ_CONFIGS = [2, 10, 20, 30]  # 2x2 grid
    grid_layout = [(0, 0), (0, 1), (1, 0), (1, 1)]  # For 2x2 grid

    # Load trained model
    print("\n1. Loading trained model...")
    model_path = Path("../Task1/checkpoints/final_model.pt")
    task2_analysis = Path("../Task2/logs/task2_analysis.json")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return None

    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type=None,
        d_vocab=P + 1,
        d_vocab_out=P,
        n_ctx=3,
        device=device,
        seed=999,
    )
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")

    # Load key frequencies
    print("\n2. Loading key frequencies from Task 2...")
    with open(task2_analysis, 'r') as f:
        task2_data = json.load(f)

    all_key_frequencies = torch.tensor(task2_data['key_frequencies'], dtype=torch.long).to(device)
    print(f"✓ Loaded {len(all_key_frequencies)} key frequencies")

    # Load dataset
    print("\n3. Loading dataset...")
    dataset = ModularAdditionDataset(p=P, frac=0.3, seed=999)
    test_inputs, test_targets = dataset.get_test_data()
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    # Create 2x2 grid plot
    print("\n4. Creating 2x2 grid of PCA circles...")
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    ablation = AblationExperiments(model, p=P, device=device)
    pca_analyzer = EmbeddingPCAAnalysis(model, p=P, device=device)

    results_grid = {}

    for idx, (num_freqs, (row, col)) in enumerate(zip(FREQ_CONFIGS, grid_layout)):
        print(f"\n   Processing {num_freqs} frequencies ({idx+1}/4)...")

        ax = fig.add_subplot(gs[row, col])

        # Use top-num_freqs frequencies
        top_k_freqs = all_key_frequencies[:num_freqs]

        # Fit PCA
        embeddings_2d, explained_var = pca_analyzer.fit_pca(n_components=2)

        # Get PCA coordinates (only numeric tokens, exclude '=')
        pca_coords = embeddings_2d[:P]
        tokens = np.arange(P)

        # Compute circularity metrics
        circularity = pca_analyzer.compute_circularity_metrics()

        # Run ablation
        ablation_results = ablation.run_ablation_sweep(test_inputs, test_targets, top_k_freqs)
        baseline_acc = ablation_results['baseline_accuracy']
        mean_drop = float(np.mean([r['accuracy_drop'] for r in ablation_results['ablation_results']]))

        # Run inverse ablation
        inverse_ablation_results = ablation.run_inverse_ablation(test_inputs, test_targets, top_k_freqs)
        inverse_acc = inverse_ablation_results['accuracy_with_only_key_frequencies']

        # Create scatter plot on this subplot
        scatter = ax.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            c=tokens,
            s=100,
            cmap='hsv',
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

        ax.set_xlabel('PC1', fontsize=11)
        ax.set_ylabel('PC2', fontsize=11)

        # Title with key metrics
        title_text = f"{num_freqs} Frequencies\n"
        title_text += f"Baseline: {baseline_acc:.3f} | Inverse: {inverse_acc:.3f}\n"
        title_text += f"CV: {circularity['cv_distance']:.4f} | Circular: {circularity['is_circular']}"
        ax.set_title(title_text, fontsize=12, fontweight='bold')

        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

        # Add colorbar for this subplot
        cbar = plt.colorbar(scatter, ax=ax, label='Token Value')

        results_grid[num_freqs] = {
            'baseline_accuracy': baseline_acc,
            'inverse_accuracy': inverse_acc,
            'mean_ablation_drop': mean_drop,
            'circularity_cv': circularity['cv_distance'],
            'is_circular': circularity['is_circular'],
            'explained_variance_pc1': float(explained_var[0]),
            'explained_variance_pc2': float(explained_var[1]),
        }

    # Add overall title
    fig.suptitle(
        'Frequency Count Comparison: PCA Circular Representations\n' +
        'Each subplot shows embeddings for different numbers of key frequencies',
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    plot_path = TASK_DIR / "plots" / "frequency_comparison_pca_grid.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {plot_path}")

    # Save results
    results_path = TASK_DIR / "logs" / "frequency_comparison_grid_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_grid, f, indent=2, default=str)
    print(f"✓ Saved: {results_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("FREQUENCY CONFIGURATION COMPARISON")
    print(f"{'='*80}\n")
    print(f"{'Freq':<6} {'Baseline':<12} {'Inverse':<12} {'Mean Drop':<12} {'CV':<10} {'Circular':<10}")
    print("-" * 80)

    for num_freqs in FREQ_CONFIGS:
        data = results_grid[num_freqs]
        print(f"{num_freqs:<6} {data['baseline_accuracy']:<12.4f} {data['inverse_accuracy']:<12.4f} "
              f"{data['mean_ablation_drop']:<12.4f} {data['circularity_cv']:<10.4f} "
              f"{str(data['is_circular']):<10}")

    return results_grid


if __name__ == "__main__":
    results = main()