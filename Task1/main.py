"""Task 1: Setup and reproduce the grokking phenomenon."""
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import torch
from pathlib import Path
import json

from dataset import ModularAdditionDataset
from train import ModularAdditionTrainer
from loss import compute_accuracy
from utils.common import set_seed, get_device, log_message, create_log_file
from utils.viz_utils import plot_training_curves

try:
    from transformer_lens import HookedTransformerConfig, HookedTransformer
except ImportError:
    print("ERROR: TransformerLens not installed. Install with: pip install transformer-lens")
    sys.exit(1)


def create_model(device: torch.device, p: int = 113):
    """Create HookedTransformer for modular addition."""
    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=4,
        d_model=128,
        d_head=32,
        d_mlp=512,
        act_fn="relu",
        normalization_type=None,
        d_vocab=p + 1,  # +1 for '=' token
        d_vocab_out=p,
        n_ctx=3,  # Context length: [a, b, =]
        init_weights=True,
        device=device,
        seed=999,
    )
    model = HookedTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model created with {n_params} parameters")
    return model


def main():
    """Main training pipeline."""
    # Setup
    TASK_DIR = Path(".")
    set_seed(999)
    device = get_device()
    log_file = create_log_file(TASK_DIR)

    print("="*80)
    print("TASK 1: SETUP & REPRODUCTION - GROKKING ON MODULAR ADDITION")
    print("="*80)

    # Hyperparameters
    P = 113
    FRAC = 0.3
    NUM_EPOCHS = 50000  
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 2.0
    BETA1 = 0.9
    BETA2 = 0.98

    # Checkpoint to continue from
    CHECKPOINT_PATH = Path("checkpoints/final_model.pt")
    CONTINUE_FROM_CHECKPOINT = CHECKPOINT_PATH.exists()

    log_message(f"Modulus P: {P}", log_file)
    log_message(f"Training fraction: {FRAC}", log_file)
    log_message(f"Additional epochs: {NUM_EPOCHS}", log_file)
    log_message(f"Learning rate: {LEARNING_RATE}", log_file)
    log_message(f"Weight decay: {WEIGHT_DECAY}", log_file)

    if CONTINUE_FROM_CHECKPOINT:
        print(f"✓ Found checkpoint at {CHECKPOINT_PATH}")
        log_message(f"Continuing from checkpoint: {CHECKPOINT_PATH}", log_file)
    else:
        print(f"✗ No checkpoint found at {CHECKPOINT_PATH}")
        log_message("WARNING: No checkpoint found, starting from scratch", log_file)

    # Create dataset
    print("\n1. Creating dataset...")
    log_message("Creating dataset", log_file)
    dataset = ModularAdditionDataset(p=P, frac=FRAC, seed=999)

    BATCH_SIZE = len(dataset.train_pairs)  # Full-batch training (required for grokking)

    train_loader = dataset.get_full_train_loader(batch_size=BATCH_SIZE)
    test_loader = dataset.get_full_test_loader(batch_size=BATCH_SIZE)
    log_message(f"Dataset created: {len(dataset.train_pairs)} train, {len(dataset.test_pairs)} test", log_file)

    # Create model
    print("\n2. Creating model...")
    log_message("Creating HookedTransformer", log_file)
    model = create_model(device, p=P)
    model = model.to(device)

    # Create trainer
    print("\n3. Initializing trainer...")
    log_message("Initializing trainer", log_file)
    trainer = ModularAdditionTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        beta1=BETA1,
        beta2=BETA2,
    )

    # Load checkpoint if available
    if CONTINUE_FROM_CHECKPOINT:
        print("\n3b. Loading checkpoint...")
        log_message("Loading model from checkpoint", log_file)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"✓ Model state loaded from {CHECKPOINT_PATH}")
            log_message(f"Model state loaded successfully", log_file)
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            log_message(f"ERROR loading checkpoint: {e}", log_file)

    # Freeze biases
    print("\n4. Freezing biases...")
    log_message("Freezing biases", log_file)
    trainer.freeze_biases()

    # Train
    print("\n5. Starting training...")
    log_message(f"Starting training for {NUM_EPOCHS} epochs", log_file)
    print("-" * 80)

    metrics = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        checkpoint_dir=str(TASK_DIR / "checkpoints"),
    )

    print("-" * 80)

    # Save final model
    print("\n6. Saving model and metrics...")
    log_message("Saving model and metrics", log_file)
    model_path = TASK_DIR / "checkpoints" / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    log_message(f"Model saved to {model_path}", log_file)

    metrics_path = TASK_DIR / "logs" / "metrics.json"
    trainer.save_metrics(str(metrics_path))
    log_message(f"Metrics saved to {metrics_path}", log_file)

    # Generate plot (Figure 2 style)
    print("\n7. Generating training curves...")
    log_message("Generating training curves plot", log_file)

    plot_data = {
        'train': metrics['train_loss'],
        'test': metrics['test_loss'],
    }
    acc_data = {
        'train': metrics['train_acc'],
        'test': metrics['test_acc'],
    }

    plot_path = TASK_DIR / "plots" / "figure2_training_curves.png"
    plot_training_curves(plot_data, acc_data, str(plot_path), title="Figure 2: Grokking on Modular Addition")
    log_message(f"Plot saved to {plot_path}", log_file)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if CONTINUE_FROM_CHECKPOINT:
        print("✓ Continuation run (loaded from checkpoint)")
        print(f"  Previous run: {NUM_EPOCHS} epochs with LR={LEARNING_RATE}, WD={WEIGHT_DECAY}")
    print(f"Final training loss:  {metrics['train_loss'][-1]:.4f}")
    print(f"Final training acc:   {metrics['train_acc'][-1]:.4f}")
    print(f"Final test loss:      {metrics['test_loss'][-1]:.4f}")
    print(f"Final test accuracy:  {metrics['test_acc'][-1]:.4f}")
    print(f"Total epochs trained: {len(metrics['epochs'])}")
    print(f"\nModel saved to:       {model_path}")
    print(f"Metrics saved to:     {metrics_path}")
    print(f"Plot saved to:        {plot_path}")
    print(f"Logs saved to:        {log_file}")

    log_message("Training completed successfully!", log_file)
    log_message(f"Final test accuracy: {metrics['test_acc'][-1]:.4f}", log_file)

    return model, metrics, dataset


if __name__ == "__main__":
    model, metrics, dataset = main()
