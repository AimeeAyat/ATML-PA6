"""Ablation and intervention experiments on trained model."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from pathlib import Path
import json


class AblationExperiments:
    """Perform ablation experiments on trained model."""

    def __init__(self, model, p: int = 113, device: torch.device = None):
        """
        Initialize ablation.

        Args:
            model: HookedTransformer model
            p: Modulus
            device: torch.device
        """
        self.model = model
        self.p = p
        self.device = device or torch.device("cpu")
        self.original_state = None

    def save_original_state(self):
        """Save original model state for restoration."""
        self.original_state = {name: param.clone() for name, param in self.model.named_parameters()}

    def restore_original_state(self):
        """Restore model to original state."""
        if self.original_state is None:
            raise ValueError("Original state not saved")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_state:
                    param.copy_(self.original_state[name])

    def ablate_fourier_component(
        self,
        frequency_idx: int,
        component_type: str = 'embedding',
    ) -> Dict:
        """
        Zero out a specific Fourier component in the embedding or unembedding matrix.

        Args:
            frequency_idx: Which frequency to ablate
            component_type: 'embedding' or 'unembedding'

        Returns:
            Dictionary with ablation results
        """
        from torch.fft import fft, ifft

        if component_type == 'embedding':
            W = self.model.embed.W_E  # [vocab, d_model]
        else:
            W = self.model.unembed.W_U  # [vocab_out, d_model]

        # Compute FFT
        W_T = W.T  # [d_model, vocab]
        W_fft = fft(W_T.to(torch.float32), dim=1)

        # Zero out the specific frequency
        W_fft[:, frequency_idx] = 0

        # Transform back
        W_ablated = torch.real(ifft(W_fft, dim=1))
        W_ablated = W_ablated.T.to(W.dtype)

        # Apply ablation
        with torch.no_grad():
            if component_type == 'embedding':
                self.model.embed.W_E.copy_(W_ablated)
            else:
                self.model.unembed.W_U.copy_(W_ablated)

        results = {
            'frequency_idx': frequency_idx,
            'component_type': component_type,
            'ablated_frequency': frequency_idx,
        }

        return results

    def ablate_multiple_frequencies(
        self,
        frequency_indices: list,
        component_type: str = 'embedding',
    ) -> Dict:
        """Ablate multiple frequencies at once."""
        from torch.fft import fft, ifft

        if component_type == 'embedding':
            W = self.model.embed.W_E
        else:
            W = self.model.unembed.W_U

        W_T = W.T
        W_fft = fft(W_T.to(torch.float32), dim=1)

        # Zero out all specified frequencies
        for freq_idx in frequency_indices:
            W_fft[:, int(freq_idx)] = 0

        W_ablated = torch.real(ifft(W_fft, dim=1))
        W_ablated = W_ablated.T.to(W.dtype)

        with torch.no_grad():
            if component_type == 'embedding':
                self.model.embed.W_E.copy_(W_ablated)
            else:
                self.model.unembed.W_U.copy_(W_ablated)

        results = {
            'ablated_frequencies': [int(f) for f in frequency_indices],
            'component_type': component_type,
            'num_frequencies_ablated': len(frequency_indices),
        }

        return results

    def evaluate_accuracy(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Evaluate model accuracy on test set.

        Args:
            inputs: Input tensor [batch, seq_len]
            targets: Target indices [batch]

        Returns:
            Accuracy (float in [0, 1])
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)
            logits = logits[:, -1, :]  # [batch, vocab]

            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).float()
            accuracy = correct.mean().item()

        return accuracy

    def run_ablation_sweep(
        self,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        key_frequencies: torch.Tensor,
    ) -> Dict:
        """
        Sweep through ablating each key frequency one by one.

        Args:
            test_inputs: Test input tensor
            test_targets: Test target tensor
            key_frequencies: Indices of key frequencies

        Returns:
            Dictionary with ablation results for each frequency
        """
        results = {
            'baseline_accuracy': None,
            'ablation_results': [],
        }

        # Save original state before any modifications
        self.save_original_state()

        # Get baseline
        self.restore_original_state()
        baseline_acc = self.evaluate_accuracy(test_inputs, test_targets)
        results['baseline_accuracy'] = baseline_acc
        print(f"Baseline accuracy: {baseline_acc:.4f}")

        # Ablate each frequency
        for freq_idx in key_frequencies:
            self.restore_original_state()
            self.ablate_fourier_component(int(freq_idx.item()), component_type='embedding')

            acc = self.evaluate_accuracy(test_inputs, test_targets)

            results['ablation_results'].append({
                'ablated_frequency': int(freq_idx.item()),
                'accuracy_after_ablation': acc,
                'accuracy_drop': baseline_acc - acc,
            })

            print(f"  Ablated freq {int(freq_idx.item())}: {acc:.4f} (drop: {baseline_acc - acc:.4f})")

        self.restore_original_state()
        return results

    def run_inverse_ablation(
        self,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        key_frequencies: torch.Tensor,
    ) -> Dict:
        """
        Keep ONLY key frequencies, ablate everything else.

        Args:
            test_inputs: Test input tensor
            test_targets: Test target tensor
            key_frequencies: Indices of key frequencies

        Returns:
            Dictionary with inverse ablation results
        """
        self.save_original_state()

        from torch.fft import fft, ifft

        W = self.model.embed.W_E
        W_T = W.T
        W_fft = fft(W_T.to(torch.float32), dim=1)

        # Create mask for key frequencies
        mask = torch.zeros_like(W_fft)
        for freq_idx in key_frequencies:
            mask[:, int(freq_idx.item())] = 1

        # Keep only key frequencies
        W_fft_filtered = W_fft * mask

        W_reconstructed = torch.real(ifft(W_fft_filtered, dim=1))
        W_reconstructed = W_reconstructed.T.to(W.dtype)

        with torch.no_grad():
            self.model.embed.W_E.copy_(W_reconstructed)

        # Evaluate
        baseline_acc = None
        self.restore_original_state()
        baseline_acc = self.evaluate_accuracy(test_inputs, test_targets)

        self.save_original_state()
        self.restore_original_state()
        self.ablate_multiple_frequencies(
            [i for i in range(self.p) if i not in key_frequencies.tolist()],
            component_type='embedding'
        )

        inverse_ablated_acc = self.evaluate_accuracy(test_inputs, test_targets)

        self.restore_original_state()

        results = {
            'baseline_accuracy': baseline_acc,
            'accuracy_with_only_key_frequencies': inverse_ablated_acc,
            'num_key_frequencies': len(key_frequencies),
            'num_total_frequencies': self.p,
            'sparsity_ratio': len(key_frequencies) / self.p,
            'accuracy_preserved': inverse_ablated_acc / baseline_acc if baseline_acc > 0 else 0,
        }

        return results
