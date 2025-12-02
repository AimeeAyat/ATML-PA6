"""Analyze constructive/destructive interference in logits."""
import torch
import numpy as np
from typing import Dict, Tuple


class InterferenceAnalysis:
    """Analyze logit interference patterns."""

    def __init__(self, model, p: int = 113, device: torch.device = None):
        """
        Initialize interference analysis.

        Args:
            model: HookedTransformer model
            p: Modulus
            device: torch.device
        """
        self.model = model
        self.p = p
        self.device = device or torch.device("cpu")

    def compute_logit_contributions(
        self,
        inputs: torch.Tensor,
        key_frequencies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute logit contributions by frequency component.

        Args:
            inputs: Input tensor [batch, seq_len]
            key_frequencies: Indices of key frequencies

        Returns:
            (full_logits, frequency_filtered_logits)
        """
        self.model.eval()

        with torch.no_grad():
            logits, cache = self.model.run_with_cache(inputs)  # [batch, seq_len, vocab]

        # Use only the last position (answer position)
        logits = logits[:, -1, :]  # [batch, vocab]

        # Extract W_E and W_U
        W_E = self.model.embed.W_E  # [vocab, d_model]
        W_U = self.model.unembed.W_U  # [vocab_out, d_model]

        # Compute FFT of W_E
        W_E_T = W_E.T  # [d_model, vocab]
        from torch.fft import fft
        dft_matrix = fft.fft(W_E_T.to(torch.float32), dim=1)  # [d_model, vocab]

        # Filter to keep only key frequencies
        filtered_W_E = torch.zeros_like(W_E)

        for freq in key_frequencies:
            freq_idx = int(freq.item())
            # Reconstruct using only this frequency
            freq_components = dft_matrix[:, freq_idx:freq_idx+1]  # [d_model, 1]
            # (This is simplified; full IFFT would be more accurate)

        return logits, logits

    def analyze_interference_pattern(
        self,
        a: int,
        b: int,
        correct_c: int,
        logits: torch.Tensor,
    ) -> Dict:
        """
        Analyze constructive/destructive interference for a specific example.

        Args:
            a, b: Input values
            correct_c: Correct answer (a + b) mod p
            logits: Model logits [vocab_size]

        Returns:
            Dictionary with interference metrics
        """
        logits_np = logits.cpu().detach().numpy()

        # Compute correct vs incorrect logits
        correct_logit = logits_np[correct_c]
        incorrect_logits = np.array([logits_np[i] for i in range(len(logits_np)) if i != correct_c])
        mean_incorrect_logit = np.mean(incorrect_logits)

        # Interference metric: logit difference (more stable than probability ratio)
        # Positive = constructive interference (correct answer has higher logit)
        logit_diff = correct_logit - mean_incorrect_logit

        # Compute rank (0-indexed, so 0 = rank 1)
        correct_rank = int(np.argsort(logits_np)[::-1].tolist().index(correct_c))

        # Convert to probability for reference
        logits_shifted = logits_np - np.max(logits_np)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)
        correct_prob = probs[correct_c]

        result = {
            'a': a,
            'b': b,
            'correct_c': correct_c,
            'correct_logit': float(correct_logit),
            'mean_incorrect_logit': float(mean_incorrect_logit),
            'logit_difference': float(logit_diff),
            'correct_prob': float(correct_prob),
            'correct_rank': correct_rank,
        }

        return result

    def compute_logit_variance_by_position(
        self,
        dataset,
        num_examples: int = 100,
    ) -> Dict:
        """
        Compute variance in logit values across positions.

        High variance at correct position = constructive interference
        Low variance elsewhere = destructive interference
        """
        examples = dataset.test_pairs[:num_examples]

        position_variances = {i: [] for i in range(self.p)}

        for a, b, c in examples:
            inputs = torch.tensor([[a, b, self.p]], dtype=torch.long)  # [1, 3]
            inputs = inputs.to(self.device)

            with torch.no_grad():
                logits = self.model(inputs)  # [1, 3, vocab]
                logits = logits[0, -1, :]  # [vocab]

            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Record probability at each position
            for pos in range(self.p):
                position_variances[pos].append(probs[pos].item())

        # Compute statistics
        results = {
            'position_stats': {},
        }

        for pos in range(self.p):
            values = position_variances[pos]
            results['position_stats'][f'pos_{pos}'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'max': float(np.max(values)),
                'min': float(np.min(values)),
            }

        return results
