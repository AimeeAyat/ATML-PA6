"""Task 2: Reverse engineering the Fourier Circuit."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Task1'))

import torch
from pathlib import Path
import json

from fourier_analysis import FourierCircuitAnalysis
from interference_analysis import InterferenceAnalysis
from utils.common import set_seed, get_device, log_message, create_log_file
from utils.viz_utils import plot_fourier_spectrum, plot_heatmap, plot_logit_interference

try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("ERROR: TransformerLens not installed")
    sys.exit(1)

# Add Task1 to path for dataset
sys.path.insert(0, os.path.abspath('../Task1'))
from dataset import ModularAdditionDataset


def main():
    """Main analysis pipeline."""
    TASK_DIR = Path(".")
    set_seed(999)
    device = get_device()
    log_file = create_log_file(TASK_DIR)

    print("="*80)
    print("TASK 2: REVERSE ENGINEERING THE FOURIER CIRCUIT")
    print("="*80)

    P = 113

    # Load trained model from Task 1
    print("\n1. Loading trained model from Task 1...")
    log_message("Loading trained model", log_file)

    model_path = Path("../Task1/checkpoints/final_model.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run Task 1 first!")
        return None

    try:
        from transformer_lens import HookedTransformerConfig, HookedTransformer

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
            init_weights=True,
            device=device,
            seed=999,
        )
        model = HookedTransformer(cfg)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"Model loaded from {model_path}")
        log_message(f"Model loaded successfully", log_file)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None

    # Load dataset
    print("\n2. Loading dataset...")
    log_message("Loading dataset", log_file)
    dataset = ModularAdditionDataset(p=P, frac=0.3, seed=999)

    # Initialize Fourier analysis
    print("\n3. Analyzing Fourier spectrum...")
    log_message("Starting Fourier analysis", log_file)

    fourier_analyzer = FourierCircuitAnalysis(model, p=P, device=device)

    # Compute DFT
    dft_matrix, fourier_norms = fourier_analyzer.compute_dft_embedding()
    log_message(f"DFT computed: shape {dft_matrix.shape}", log_file)

    # Identify key frequencies
    key_frequencies = fourier_analyzer.identify_key_frequencies(fourier_norms, top_k=10)
    print(f"\nKey frequencies identified: {key_frequencies.cpu().numpy()}")
    log_message(f"Key frequencies: {key_frequencies.cpu().numpy()}", log_file)

    # Plot Fourier spectrum
    print("\n4. Plotting Fourier spectrum...")
    plot_path = TASK_DIR / "plots" / "fourier_spectrum.png"
    plot_fourier_spectrum(fourier_norms, str(plot_path), title="Fourier Spectrum of Embedding Matrix")
    log_message(f"Fourier spectrum plot saved", log_file)

    # Analyze sparsity
    print("\n5. Analyzing sparsity...")
    sparsity_analysis = fourier_analyzer.analyze_sparsity(fourier_norms)
    print(f"Gini coefficient: {sparsity_analysis['gini_coefficient']:.4f}")
    print(f"Energy concentration (top-10): {sparsity_analysis['energy_concentration']['top_10']:.4f}")
    log_message(f"Sparsity analysis: {json.dumps(sparsity_analysis, indent=2)}", log_file)

    # Verify trigonometric identity
    print("\n6. Verifying trigonometric identity...")
    log_message("Verifying cos(ω(a+b)) = cos(ωa)cos(ωb) - sin(ωa)sin(ωb)", log_file)

    identity_results = fourier_analyzer.verify_trigonometric_identity(
        dataset,
        key_frequencies,
        num_examples=50
    )
    print(f"Mean identity error: {identity_results['mean_identity_error']:.6f}")
    print(f"Std identity error: {identity_results['std_identity_error']:.6f}")
    log_message(f"Identity verification complete: mean error {identity_results['mean_identity_error']:.6f}", log_file)

    # Analyze interference patterns
    print("\n7. Analyzing logit interference patterns...")
    log_message("Analyzing constructive/destructive interference", log_file)

    interference_analyzer = InterferenceAnalysis(model, p=P, device=device)

    # Analyze a few example cases
    example_count = 0
    interference_examples = []

    for a, b, c in dataset.test_pairs[:100]:
        inputs = torch.tensor([[a, b, P]], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(inputs)  # [1, 3, vocab]
            logits = logits[0, -1, :]  # [vocab]

        interference_result = interference_analyzer.analyze_interference_pattern(
            int(a), int(b), int(c), logits
        )
        interference_examples.append(interference_result)
        example_count += 1

        if example_count == 5:
            # Plot logit interference for first example
            if example_count == 1:
                plot_path = TASK_DIR / "plots" / "logit_interference_example.png"
                plot_logit_interference(logits, int(c), str(plot_path), title=f"Logit Interference: {a} + {b} = {c}")

    # Compute statistics
    correct_ranks = [ex['correct_rank'] for ex in interference_examples]
    logit_diffs = [ex['logit_difference'] for ex in interference_examples]

    print(f"\nCorrect answer rank (mean): {sum(correct_ranks)/len(correct_ranks):.2f}")
    print(f"Logit difference (mean): {sum(logit_diffs)/len(logit_diffs):.2f}")

    log_message(
        f"Interference analysis: mean correct rank {sum(correct_ranks)/len(correct_ranks):.2f}",
        log_file
    )

    # Save analysis results
    print("\n8. Saving analysis results...")
    analysis_results = {
        'fourier_norms': fourier_norms.detach().cpu().numpy().tolist(),
        'key_frequencies': key_frequencies.detach().cpu().numpy().tolist(),
        'sparsity_analysis': sparsity_analysis,
        'identity_verification': {
            'mean_error': identity_results['mean_identity_error'],
            'std_error': identity_results['std_identity_error'],
        },
        'interference_analysis': {
            'mean_correct_rank': float(sum(correct_ranks) / len(correct_ranks)),
            'mean_logit_difference': float(sum(logit_diffs) / len(logit_diffs)),
            'examples': interference_examples[:10],
        },
    }

    analysis_path = TASK_DIR / "logs" / "task2_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"Analysis saved to {analysis_path}")
    log_message("Analysis results saved", log_file)

    print("\n" + "="*80)
    print("TASK 2 COMPLETE")
    print("="*80)
    print(f"Key findings:")
    print(f"  - Gini coefficient (sparsity): {sparsity_analysis['gini_coefficient']:.4f}")
    print(f"  - Top-10 energy concentration: {sparsity_analysis['energy_concentration']['top_10']:.4f}")
    print(f"  - Identity verification error: {identity_results['mean_identity_error']:.6f}")
    print(f"  - Mean correct answer rank: {sum(correct_ranks) / len(correct_ranks):.2f} (lower is better)")
    print(f"  - Mean logit difference: {sum(logit_diffs) / len(logit_diffs):.2f} (higher is better)")

    return model, analysis_results


if __name__ == "__main__":
    model, analysis_results = main()
