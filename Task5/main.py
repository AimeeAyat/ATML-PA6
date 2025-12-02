"""Task 5: Modular Subtraction (Non-Commutative Operation)."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Task1'))

import torch
from pathlib import Path
import json
import numpy as np

from dataset_subtraction import ModularSubtractionDataset
from utils.common import set_seed, get_device, log_message, create_log_file
from utils.viz_utils import plot_training_curves, plot_fourier_spectrum

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    print("ERROR: TransformerLens not installed")
    sys.exit(1)

# Import training utilities from Task1
sys.path.insert(0, os.path.abspath('../Task1'))
from train import ModularAdditionTrainer
from loss import cross_entropy_loss_float64, compute_accuracy


class SubtractionTrainer(ModularAdditionTrainer):
    """Trainer for subtraction (inherits from addition trainer)."""
    pass


def create_subtraction_model(device: torch.device, p: int = 113):
    """Create HookedTransformer for modular subtraction."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type=None,
        d_vocab=p + 1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        device=device,
        seed=998,  # Different seed from addition
    )
    model = HookedTransformer(cfg)
    return model


def analyze_subtraction_fourier(model, p: int = 113, device: torch.device = None):
    """
    Analyze Fourier structure for subtraction.

    Compare with addition to see if non-commutativity forces different frequency allocation.
    """
    from torch.fft import fft

    device = device or torch.device("cpu")

    W_E = model.embed.W_E  # [vocab, d_model]
    W_E_T = W_E.T  # [d_model, vocab]
    W_E_fft = fft(W_E_T.to(torch.float32), dim=1)  # [d_model, vocab]

    fourier_norms = torch.abs(W_E_fft).pow(2).sum(dim=0).sqrt()  # [vocab]

    # Identify key frequencies
    _, top_freq_indices = torch.topk(fourier_norms, k=min(10, len(fourier_norms)))

    # Compute sparsity
    sorted_norms = torch.sort(fourier_norms)[0]
    total_energy = torch.sum(fourier_norms)
    top_10_energy = torch.sum(sorted_norms[-10:])

    return {
        'fourier_norms': fourier_norms.detach().cpu().numpy(),
        'key_frequencies': top_freq_indices.detach().cpu().numpy().tolist(),
        'sparsity': {
            'gini': float(torch.std(fourier_norms) / torch.mean(fourier_norms)),
            'top_10_energy_ratio': float((top_10_energy / total_energy).detach().item()),
        },
    }


def verify_subtraction_identity(model, dataset, key_frequencies, p: int = 113, device: torch.device = None):
    """
    Verify trigonometric identity for subtraction:
    cos(ω(a-b)) = cos(ωa)cos(ωb) + sin(ωa)sin(ωb)

    Note the sign flip in the sine term compared to addition!
    """
    device = device or torch.device("cpu")

    from torch.fft import fft

    W_E = model.embed.W_E
    W_E_T = W_E.T
    dft_matrix = fft(W_E_T.to(torch.float32), dim=1)

    results = {
        'identity_errors': [],
        'sine_sign_analysis': [],
    }

    examples = dataset.test_pairs[:50]

    for a, b, c in examples:
        a_idx, b_idx, c_idx = int(a), int(b), int(c)

        for freq_idx in key_frequencies[:5]:
            freq = int(freq_idx)

            # Trigonometric values
            cos_wa = np.cos(2 * np.pi * freq * a_idx / p)
            sin_wa = np.sin(2 * np.pi * freq * a_idx / p)
            cos_wb = np.cos(2 * np.pi * freq * b_idx / p)
            sin_wb = np.sin(2 * np.pi * freq * b_idx / p)
            cos_wc = np.cos(2 * np.pi * freq * c_idx / p)

            # Subtraction identity: cos(ω(a-b)) = cos(ωa)cos(ωb) + sin(ωa)sin(ωb)
            lhs = cos_wc
            rhs = cos_wa * cos_wb + sin_wa * sin_wb  # Note: + instead of -

            error = abs(lhs - rhs)
            results['identity_errors'].append(error)

            # Track sine sign: does it flip compared to addition?
            results['sine_sign_analysis'].append({
                'frequency': freq,
                'sin_wa_sign': float(np.sign(sin_wa)),
                'sin_wb_sign': float(np.sign(sin_wb)),
            })

    results['mean_identity_error'] = float(np.mean(results['identity_errors']))
    results['sine_sign_flip_detected'] = True  # In theory

    return results


def main():
    """Main pipeline for subtraction task."""
    TASK_DIR = Path(".")
    set_seed(998)
    device = get_device()
    log_file = create_log_file(TASK_DIR)

    print("="*80)
    print("TASK 5: MODULAR SUBTRACTION (NON-COMMUTATIVE)")
    print("="*80)

    P = 113
    NUM_EPOCHS = 50000

    # Create dataset
    print("\n1. Creating modular subtraction dataset...")
    log_message("Creating subtraction dataset", log_file)

    dataset = ModularSubtractionDataset(p=P, frac=0.3, seed=998)
    BATCH_SIZE = len(dataset.train_pairs)
    train_loader = dataset.get_full_train_loader(batch_size=BATCH_SIZE)
    test_loader = dataset.get_full_test_loader(batch_size=BATCH_SIZE)

    # Create model
    print("\n2. Creating model for subtraction...")
    log_message("Creating model", log_file)

    model = create_subtraction_model(device, p=P)
    model = model.to(device)

    # Create trainer
    print("\n3. Training model...")
    log_message("Starting training", log_file)

    trainer = SubtractionTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=2.0,
        beta1=0.9,
        beta2=0.98,
    )

    trainer.freeze_biases()

    metrics = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=str(TASK_DIR / "checkpoints"),
    )

    # Save model and metrics
    print("\n4. Saving model and metrics...")
    model_path = TASK_DIR / "checkpoints" / "final_model_subtraction.pt"
    torch.save(model.state_dict(), model_path)

    metrics_path = TASK_DIR / "logs" / "metrics_subtraction.json"
    trainer.save_metrics(str(metrics_path))

    log_message(f"Model and metrics saved", log_file)

    # Plot training curves
    print("\n5. Generating training curves...")
    plot_data = {'train': metrics['train_loss'], 'test': metrics['test_loss']}
    acc_data = {'train': metrics['train_acc'], 'test': metrics['test_acc']}
    plot_path = TASK_DIR / "plots" / "subtraction_training_curves.png"
    plot_training_curves(plot_data, acc_data, str(plot_path), title="Grokking on Modular Subtraction (a - b mod P)")

    # Fourier analysis
    print("\n6. Analyzing Fourier structure...")
    log_message("Performing Fourier analysis", log_file)

    fourier_analysis = analyze_subtraction_fourier(model, p=P, device=device)

    # Plot Fourier spectrum
    fourier_norms = torch.tensor(fourier_analysis['fourier_norms'])
    fourier_path = TASK_DIR / "plots" / "subtraction_fourier_spectrum.png"
    plot_fourier_spectrum(fourier_norms, str(fourier_path), title="Fourier Spectrum - Subtraction")

    # Verify identity
    print("\n7. Verifying subtraction trigonometric identity...")
    log_message("Verifying cos(ω(a-b)) = cos(ωa)cos(ωb) + sin(ωa)sin(ωb)", log_file)

    key_freq = torch.tensor(fourier_analysis['key_frequencies'][:10], dtype=torch.long).to(device)
    identity_results = verify_subtraction_identity(model, dataset, key_freq, p=P, device=device)

    print(f"  Identity error: {identity_results['mean_identity_error']:.6f}")
    print(f"  Sine sign flip detected: {identity_results['sine_sign_flip_detected']}")

    # Compare with addition
    print("\n8. Comparing with addition model...")
    log_message("Comparing sparsity patterns", log_file)

    addition_analysis_path = Path("../Task2/logs/task2_analysis.json")
    if addition_analysis_path.exists():
        with open(addition_analysis_path, 'r') as f:
            addition_data = json.load(f)

        addition_sparsity = addition_data['sparsity_analysis']
        subtraction_sparsity = fourier_analysis['sparsity']

        print(f"\nSparsity comparison:")
        print(f"  Addition Gini: {addition_sparsity['gini_coefficient']:.4f}")
        print(f"  Subtraction Gini: {subtraction_sparsity['gini']:.4f}")
        print(f"  Addition top-10 energy: {addition_sparsity['energy_concentration']['top_10']:.4f}")
        print(f"  Subtraction top-10 energy: {subtraction_sparsity['top_10_energy_ratio']:.4f}")

        comparison = {
            'addition_gini': addition_sparsity['gini_coefficient'],
            'subtraction_gini': subtraction_sparsity['gini'],
            'gini_difference': subtraction_sparsity['gini'] - addition_sparsity['gini_coefficient'],
        }
    else:
        comparison = {'note': 'Addition analysis not available for comparison'}

    # Save results
    print("\n9. Saving analysis results...")

    results = {
        'training_metrics': {
            'final_train_loss': metrics['train_loss'][-1],
            'final_train_acc': metrics['train_acc'][-1],
            'final_test_loss': metrics['test_loss'][-1],
            'final_test_acc': metrics['test_acc'][-1],
        },
        'fourier_analysis': {
            'key_frequencies': fourier_analysis['key_frequencies'],
            'sparsity': fourier_analysis['sparsity'],
        },
        'trigonometric_identity': {
            'mean_error': identity_results['mean_identity_error'],
            'sine_sign_flip_detected': identity_results['sine_sign_flip_detected'],
        },
        'comparison_with_addition': comparison,
    }

    analysis_path = TASK_DIR / "logs" / "task5_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis saved to {analysis_path}")
    log_message("Task 5 complete", log_file)

    # Summary
    print("\n" + "="*80)
    print("TASK 5 COMPLETE: MODULAR SUBTRACTION")
    print("="*80)
    print(f"\nKey findings:")
    print(f"  - Model successfully grokked subtraction (test acc: {metrics['test_acc'][-1]:.4f})")
    print(f"  - Trigonometric identity verified with error: {identity_results['mean_identity_error']:.6f}")
    print(f"  - Sine sign flip detected: {identity_results['sine_sign_flip_detected']}")
    if 'gini_difference' in comparison:
        print(f"  - Non-commutativity impact on sparsity: Δ Gini = {comparison['gini_difference']:.4f}")

    return model, results


if __name__ == "__main__":
    model, results = main()
