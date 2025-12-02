"""Frequency scaling analysis - test ablation with different numbers of key frequencies."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Task1'))

import torch
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from ablation import AblationExperiments
from utils.common import set_seed, get_device, log_message, create_log_file
from dataset import ModularAdditionDataset

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError as e:
    print(f"WARNING: TransformerLens import error: {e}")
    print("Attempting to import from different path...")
    try:
        from transformer_lens import HookedTransformer, HookedTransformerConfig
    except:
        print("ERROR: TransformerLens not available")
        sys.exit(1)


def main():
    """Test ablation with different numbers of key frequencies."""
    TASK_DIR = Path(".")
    set_seed(999)
    device = get_device()
    log_file = create_log_file(TASK_DIR)

    print("="*80)
    print("FREQUENCY SCALING ANALYSIS")
    print("="*80)

    P = 113
    FREQ_CONFIGS = [2, 10, 20, 30]  # Test with 2, 10, 20, 30 frequencies

    # Load trained model
    print("\n1. Loading trained model...")
    log_message("Loading trained model for frequency scaling analysis", log_file)

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
    print("Model loaded successfully")

    # Load key frequencies from Task 2
    print("\n2. Loading key frequencies from Task 2...")
    with open(task2_analysis, 'r') as f:
        task2_data = json.load(f)

    all_key_frequencies = torch.tensor(task2_data['key_frequencies'], dtype=torch.long).to(device)
    print(f"Loaded {len(all_key_frequencies)} key frequencies: {all_key_frequencies.cpu().numpy().tolist()}")
    log_message(f"Key frequencies: {all_key_frequencies.cpu().numpy().tolist()}", log_file)

    # Load dataset
    print("\n3. Loading dataset...")
    dataset = ModularAdditionDataset(p=P, frac=0.3, seed=999)
    test_inputs, test_targets = dataset.get_test_data()
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)
    print(f"Dataset loaded: {len(test_inputs)} test examples")

    # Initialize ablation experiments
    print("\n4. Running frequency scaling analysis...")
    ablation = AblationExperiments(model, p=P, device=device)

    results_by_config = {}
    accuracy_drops = {}
    inverse_ablation_acc = {}

    for num_freqs in FREQ_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing with {num_freqs} key frequencies...")
        print(f"{'='*60}")

        # Use top-num_freqs frequencies
        top_k_freqs = all_key_frequencies[:num_freqs]
        print(f"Using frequencies: {top_k_freqs.cpu().numpy().tolist()}")

        # Run ablation sweep
        print(f"\nAblating individual frequencies...")
        ablation_results = ablation.run_ablation_sweep(test_inputs, test_targets, top_k_freqs)

        baseline_acc = ablation_results['baseline_accuracy']
        mean_drop = float(np.mean([r['accuracy_drop'] for r in ablation_results['ablation_results']]))
        max_drop = float(np.max([r['accuracy_drop'] for r in ablation_results['ablation_results']]))

        print(f"  Baseline accuracy: {baseline_acc:.4f}")
        print(f"  Mean accuracy drop: {mean_drop:.4f}")
        print(f"  Max accuracy drop: {max_drop:.4f}")

        accuracy_drops[num_freqs] = {
            'baseline': baseline_acc,
            'mean_drop': mean_drop,
            'max_drop': max_drop,
        }

        # Run inverse ablation
        print(f"\nRunning inverse ablation (keep only {num_freqs} frequencies)...")
        inverse_ablation_results = ablation.run_inverse_ablation(test_inputs, test_targets, top_k_freqs)

        inverse_acc = inverse_ablation_results['accuracy_with_only_key_frequencies']
        inverse_ablation_acc[num_freqs] = {
            'baseline': inverse_ablation_results['baseline_accuracy'],
            'accuracy_with_key_freqs': inverse_acc,
            'accuracy_preserved_percent': inverse_ablation_results['accuracy_preserved'] * 100,
        }

        print(f"  Accuracy with ONLY {num_freqs} frequencies: {inverse_acc:.4f}")
        print(f"  Accuracy preserved: {inverse_ablation_results['accuracy_preserved']*100:.1f}%")

        results_by_config[num_freqs] = {
            'ablation': ablation_results,
            'inverse_ablation': inverse_ablation_results,
        }

        log_message(
            f"Config {num_freqs}: baseline={baseline_acc:.4f}, "
            f"inverse_acc={inverse_acc:.4f}, preserved={inverse_ablation_results['accuracy_preserved']*100:.1f}%",
            log_file
        )

    # Create comparison plots
    print(f"\n{'='*60}")
    print("Creating comparison plots...")
    print(f"{'='*60}")

    # Plot 1: Ablation impact comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    configs = list(accuracy_drops.keys())
    baseline_accs = [accuracy_drops[c]['baseline'] for c in configs]
    mean_drops = [accuracy_drops[c]['mean_drop'] for c in configs]
    max_drops = [accuracy_drops[c]['max_drop'] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, baseline_accs, width, label='Baseline Accuracy', alpha=0.8)
    ax.bar(x, mean_drops, width, label='Mean Accuracy Drop', alpha=0.8)
    ax.bar(x + width, max_drops, width, label='Max Accuracy Drop', alpha=0.8)

    ax.set_xlabel('Number of Key Frequencies', fontsize=12)
    ax.set_ylabel('Accuracy / Drop', fontsize=12)
    ax.set_title('Ablation Impact: Effect of Frequency Count', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in configs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plot_path = TASK_DIR / "plots" / "frequency_scaling_ablation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

    # Plot 2: Inverse ablation comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    baseline_accs_inv = [inverse_ablation_acc[c]['baseline'] for c in configs]
    inverse_accs = [inverse_ablation_acc[c]['accuracy_with_key_freqs'] for c in configs]
    preserved_pcts = [inverse_ablation_acc[c]['accuracy_preserved_percent'] for c in configs]

    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, baseline_accs_inv, width, label='Baseline Accuracy', alpha=0.8)
    ax.bar(x, inverse_accs, width, label='Accuracy with Only Key Freqs', alpha=0.8)
    ax.bar(x + width, [p/100 for p in preserved_pcts], width, label='Preserved %', alpha=0.8)

    ax.set_xlabel('Number of Key Frequencies', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Inverse Ablation: Accuracy Preserved vs Frequency Count', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}' for c in configs])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

    plot_path = TASK_DIR / "plots" / "frequency_scaling_inverse.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

    # Plot 3: Sparsity vs Accuracy
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sparsity_ratios = [c / P * 100 for c in configs]  # % of frequencies
    inverse_accs = [inverse_ablation_acc[c]['accuracy_with_key_freqs'] for c in configs]

    ax.plot(sparsity_ratios, inverse_accs, 'o-', linewidth=2, markersize=10, color='#2E86AB')
    ax.fill_between(sparsity_ratios, inverse_accs, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Sparsity (% of Frequencies Used)', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Sparsity-Accuracy Trade-off: Minimal Frequencies for Circuit Function', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.1])

    # Add annotations
    for i, (sr, acc) in enumerate(zip(sparsity_ratios, inverse_accs)):
        ax.annotate(f'{configs[i]} freqs\n{acc:.2%}',
                   xy=(sr, acc), xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    plot_path = TASK_DIR / "plots" / "sparsity_accuracy_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {plot_path}")

    # Save comprehensive results
    comprehensive_results = {
        'test_configurations': FREQ_CONFIGS,
        'ablation_impact': accuracy_drops,
        'inverse_ablation_accuracy': inverse_ablation_acc,
        'key_observations': {
            '2_frequencies': f"With 2 most important frequencies: {inverse_ablation_acc[2]['accuracy_preserved_percent']:.1f}% accuracy preserved",
            '10_frequencies': f"With 10 key frequencies: {inverse_ablation_acc[10]['accuracy_preserved_percent']:.1f}% accuracy preserved",
            '20_frequencies': f"With 20 frequencies: {inverse_ablation_acc[20]['accuracy_preserved_percent']:.1f}% accuracy preserved",
            '30_frequencies': f"With 30 frequencies: {inverse_ablation_acc[30]['accuracy_preserved_percent']:.1f}% accuracy preserved",
        }
    }

    results_path = TASK_DIR / "logs" / "frequency_scaling_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, default=str)
    print(f"\n✓ Saved comprehensive results: {results_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("FREQUENCY SCALING SUMMARY")
    print(f"{'='*80}\n")

    print("SPARSITY ANALYSIS:")
    for num_freqs in FREQ_CONFIGS:
        sparsity = num_freqs / P * 100
        preserved = inverse_ablation_acc[num_freqs]['accuracy_preserved_percent']
        accuracy = inverse_ablation_acc[num_freqs]['accuracy_with_key_freqs']
        print(f"  {num_freqs:2d} frequencies ({sparsity:5.1f}% of model): {accuracy:.4f} accuracy ({preserved:5.1f}% preserved)")

    print("\nABLATION SENSITIVITY:")
    for num_freqs in FREQ_CONFIGS:
        mean_drop = accuracy_drops[num_freqs]['mean_drop']
        max_drop = accuracy_drops[num_freqs]['max_drop']
        print(f"  {num_freqs:2d} frequencies: mean drop {mean_drop:.4f}, max drop {max_drop:.4f}")

    log_message("Frequency scaling analysis complete", log_file)

    return comprehensive_results


if __name__ == "__main__":
    results = main()