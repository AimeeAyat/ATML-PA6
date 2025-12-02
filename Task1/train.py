"""Training loop for modular addition task."""
import torch
import torch.optim as optim
from typing import Dict, Tuple, List
import json
from pathlib import Path
from loss import cross_entropy_loss_float64, compute_accuracy


class ModularAdditionTrainer:
    """Trainer for modular addition model."""

    def __init__(
        self,
        model,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.98,
    ):
        """
        Initialize trainer.

        Args:
            model: HookedTransformer model
            device: torch.device
            learning_rate: Learning rate (gamma = 10^-3)
            weight_decay: Weight decay (lambda = 1.0)
            beta1, beta2: Adam momentum parameters
        """
        self.model = model
        self.device = device
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epochs': [],
        }

    def freeze_biases(self):
        """Freeze all bias terms so they remain zero throughout training."""
        for name, param in self.model.named_parameters():
            if 'bias' in name:
                param.requires_grad = False
                print(f"Frozen bias: {name}")

    def train_epoch(
        self,
        train_loader,
        test_loader,
        epoch: int,
    ) -> Tuple[float, float, float, float]:
        """
        Train for one epoch and evaluate on test set.

        Returns:
            (train_loss, train_acc, test_loss, test_acc)
        """
        # Training phase
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            logits = self.model(inputs)  # [batch, seq_len, vocab]

            # Get logits for the target position (last position where answer should be)
            # Input is [a, b, =], logits are for predicting token at each position
            # We care about predicting after the = sign
            logits = logits[:, -1, :]  # [batch, vocab]

            # Compute loss
            loss = cross_entropy_loss_float64(logits, targets, reduction='mean')

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_acc += compute_accuracy(logits, targets)
            num_batches += 1

        train_loss /= num_batches
        train_acc /= num_batches

        # Test phase
        test_loss, test_acc = self.evaluate(test_loader)

        # Log metrics
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['epochs'].append(epoch)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1:5d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

        return train_loss, train_acc, test_loss, test_acc

    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                logits = logits[:, -1, :]  # [batch, vocab]

                loss = cross_entropy_loss_float64(logits, targets, reduction='mean')
                test_loss += loss.item()
                test_acc += compute_accuracy(logits, targets)
                num_batches += 1

        test_loss /= num_batches
        test_acc /= num_batches

        return test_loss, test_acc

    def train(
        self,
        train_loader,
        test_loader,
        num_epochs: int = 10000,
        checkpoint_dir: str = None,
    ):
        """
        Train model for specified number of epochs.

        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Total number of epochs
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_interval = max(1, num_epochs // 4)  # Checkpoint at 25%, 50%, 75%, 100%

        for epoch in range(num_epochs):
            train_loss, train_acc, test_loss, test_acc = self.train_epoch(
                train_loader,
                test_loader,
                epoch,
            )

            # Save checkpoint at 25% intervals
            if checkpoint_dir and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path, epoch + 1)
                print(f"Checkpoint saved: {checkpoint_path}")

        return self.metrics

    def save_checkpoint(self, filepath: str, epoch: int):
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.metrics,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']
        return checkpoint['epoch']

    def save_metrics(self, filepath: str):
        """Save metrics to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def load_metrics(self, filepath: str):
        """Load metrics from JSON."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)
