"""Task 3: Hidden Progress Measures - Detecting silent learning."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../Task1'))

import torch
from pathlib import Path
import json
import numpy as np

from progress_metrics import HiddenProgressMetrics, compute_hidden_progress_over_training
from utils.common import set_seed, get_device, log_message, create_log_file
from utils.viz_utils import plot_phase_diagram
from dataset import ModularAdditionDataset

try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
except ImportError:
    print("ERROR: TransformerLens not installed")
    sys.exit(1)


def main():
    """Main analysis pipeline for hidden progress."""
    TASK_DIR = Path(".")
    set_seed(999)
    device = get_device()
    log_file = create_log_file(TASK_DIR)

    print("="*80)
    print("TASK 3: HIDDEN PROGRESS MEASURES")
    print("="*80)

    P = 113

    # Load trained model from Task 1
    print("\n1. Loading trained model...")
    log_message("Loading trained model from Task 1", log_file)

    model_path = Path("../Task1/checkpoints/final_model.pt")
    metrics_path = Path("../Task1/logs/metrics.json")
    task2_analysis = Path("../Task2/logs/task2_analysis.json")

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run Task 1 first!")
        return None

    # Load model
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
    print("Model loaded successfully")
    log_message("Model loaded", log_file)

    # Load metrics from Task 1
    print("\n2. Loading training metrics from Task 1...")
    with open(metrics_path, 'r') as f:
        training_metrics = json.load(f)
    log_message("Training metrics loaded", log_file)

    # Load Task 2 analysis for key frequencies
    print("\n3. Loading key frequencies from Task 2...")
    with open(task2_analysis, 'r') as f:
        task2_data = json.load(f)

    key_frequencies = torch.tensor(task2_data['key_frequencies'], dtype=torch.long).to(device)
    print(f"Loaded {len(key_frequencies)} key frequencies")
    log_message(f"Loaded key frequencies: {len(key_frequencies)}", log_file)

    # Load dataset
    print("\n4. Loading dataset...")
    dataset = ModularAdditionDataset(p=P, frac=0.3, seed=999)
    log_message("Dataset loaded", log_file)

    # Compute restricted/excluded loss over training
    print("\n5. Computing restricted and excluded loss over training...")
    log_message("Computing hidden progress metrics", log_file)

    checkpoints_dir = "../Task1/checkpoints"
    progress_data = compute_hidden_progress_over_training(
        checkpoints_dir,
        dataset,
        key_frequencies,
        p=P,
        device=device,
    )

    if progress_data and len(progress_data['epochs']) > 0:
        print(f"Computed metrics for {len(progress_data['epochs'])} checkpoints")
        log_message(f"Computed hidden progress for {len(progress_data['epochs'])} checkpoints", log_file)

        # Save progress data
        progress_path = TASK_DIR / "logs" / "hidden_progress.json"
        with open(progress_path, 'w') as f:
            progress_dict = {
                'epochs': progress_data['epochs'],
                'restricted_loss': progress_data['restricted_loss'],
                'excluded_loss': progress_data['excluded_loss'],
            }
            json.dump(progress_dict, f, indent=2)
        print(f"Progress data saved to {progress_path}")
    else:
        print("No checkpoint data available for hidden progress computation")
        progress_data = None

    # Identify training phases
    print("\n6. Identifying training phases...")
    log_message("Identifying Memorization/Circuit Formation/Cleanup phases", log_file)

    metrics_computer = HiddenProgressMetrics(model, p=P, device=device)
    metrics_computer.set_key_frequencies(key_frequencies)

    phases = metrics_computer.identify_phases(training_metrics, sensitivity=0.05)
    print(f"\nPhase boundaries identified:")
    print(f"  Memorization: epochs 0-{phases['memorization'][1]}")
    print(f"  Circuit Formation: epochs {phases['circuit_formation'][0]}-{phases['circuit_formation'][1]}")
    print(f"  Cleanup: epochs {phases['cleanup'][0]}-{phases['cleanup'][1]}")
    print(f"  Grok point: epoch {phases['grok_point']}")
    log_message(f"Phase boundaries: {json.dumps({k: v for k, v in phases.items() if k != 'grok_point'}, indent=2)}", log_file)

    # Extract phase metrics
    print("\n7. Extracting metrics by phase...")
    phase_metrics = metrics_computer.extract_phase_metrics(
        [
            {
                'train_loss': training_metrics['train_loss'][i],
                'train_acc': training_metrics['train_acc'][i],
                'test_loss': training_metrics['test_loss'][i],
                'test_acc': training_metrics['test_acc'][i],
            }
            for i in range(len(training_metrics['train_loss']))
        ],
        phases
    )

    # Compute phase statistics
    phase_stats = {}
    for phase_name, phase_data in phase_metrics.items():
        if not phase_data['metrics']:
            continue

        metrics_list = phase_data['metrics']
        test_losses = [m['test_loss'] for m in metrics_list]
        test_accs = [m['test_acc'] for m in metrics_list]
        train_losses = [m['train_loss'] for m in metrics_list]

        phase_stats[phase_name] = {
            'avg_test_loss': float(np.mean(test_losses)),
            'avg_test_acc': float(np.mean(test_accs)),
            'avg_train_loss': float(np.mean(train_losses)),
            'final_test_acc': float(test_accs[-1]) if test_accs else 0,
            'num_epochs': len(metrics_list),
        }

    print("\nPhase Statistics:")
    for phase_name, stats in phase_stats.items():
        print(f"  {phase_name}:")
        print(f"    Avg test acc: {stats['avg_test_acc']:.4f}")
        print(f"    Avg test loss: {stats['avg_test_loss']:.4f}")
        print(f"    Epochs: {stats['num_epochs']}")

    log_message(f"Phase statistics: {json.dumps(phase_stats, indent=2)}", log_file)

    # Plot training curves with phase annotations
    print("\n8. Generating phase diagram...")

    phase_plot_data = {
        'epochs': list(range(len(training_metrics['train_loss']))),
        'metric1': training_metrics['test_acc'],
        'metric2': training_metrics['train_loss'],
        'phase_boundaries': [phases['memorization'][0], phases['circuit_formation'][0], phases['cleanup'][0], phases['cleanup'][1]],
    }

    phase_plot_path = TASK_DIR / "plots" / "phase_diagram.png"
    plot_phase_diagram(phase_plot_data, str(phase_plot_path), title="Training Phases: Memorization → Circuit Formation → Cleanup")
    log_message("Phase diagram plot saved", log_file)

    # Save comprehensive analysis
    print("\n9. Saving analysis results...")

    analysis_results = {
        'phases': phases,
        'phase_statistics': phase_stats,
        'key_insights': {
            'grok_epoch': phases['grok_point'],
            'final_test_accuracy': training_metrics['test_acc'][-1],
            'num_key_frequencies': int(len(key_frequencies)),
            'memorization_phase_duration': phases['circuit_formation'][0],
            'circuit_formation_duration': phases['circuit_formation'][1] - phases['circuit_formation'][0],
        }
    }

    if progress_data:
        analysis_results['hidden_progress'] = {
            'epochs': progress_data['epochs'],
            'restricted_loss': progress_data['restricted_loss'],
            'excluded_loss': progress_data['excluded_loss'],
        }

    analysis_path = TASK_DIR / "logs" / "task3_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"Analysis saved to {analysis_path}")
    log_message("Analysis results saved", log_file)

    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"\n1. GROKKING TRANSITION:")
    print(f"   - Occurs at epoch {phases['grok_point']}")
    print(f"   - Test accuracy jump from {training_metrics['test_acc'][max(0, phases['grok_point']-100)]:.4f} to {training_metrics['test_acc'][phases['grok_point']]:.4f}")

    print(f"\n2. MEMORIZATION PHASE:")
    print(f"   - Duration: {phases['circuit_formation'][0]} epochs")
    print(f"   - Avg test accuracy: {phase_stats['memorization']['avg_test_acc']:.4f}")
    print(f"   - Avg test loss: {phase_stats['memorization']['avg_test_loss']:.4f}")

    print(f"\n3. CIRCUIT FORMATION:")
    print(f"   - Duration: {phases['circuit_formation'][1] - phases['circuit_formation'][0]} epochs")
    print(f"   - Where the Fourier circuit is learned silently")
    print(f"   - Average test acc: {phase_stats['circuit_formation']['avg_test_acc']:.4f}")

    print(f"\n4. CLEANUP PHASE:")
    print(f"   - Test accuracy reaches near-perfect: {phase_stats['cleanup']['final_test_acc']:.4f}")
    print(f"   - Model sheds memorization, retains only the circuit")

    return model, analysis_results


if __name__ == "__main__":
    model, results = main()
