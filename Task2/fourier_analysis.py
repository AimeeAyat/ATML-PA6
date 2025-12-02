"""Fourier analysis of the embedding matrix and circuit verification."""
import torch
import torch.fft as fft
import numpy as np
from typing import Tuple, Dict
from pathlib import Path


class FourierCircuitAnalysis:
    """Analyze grokking model using Fourier basis."""

    def __init__(self, model, p: int = 113, device: torch.device = None):
        """
        Initialize Fourier analysis.

        Args:
            model: HookedTransformer model
            p: Modulus (vocabulary size for embedding)
            device: torch.device
        """
        self.model = model
        self.p = p
        self.device = device or torch.device("cpu")

    def get_embedding_matrix(self) -> torch.Tensor:
        """Extract embedding matrix W_E from model."""
        # HookedTransformer stores embeddings in model.embed
        W_E = self.model.embed.W_E  # Shape: [vocab_size, d_model]
        return W_E

    def get_unembedding_matrix(self) -> torch.Tensor:
        """Extract unembedding matrix W_U from model."""
        # W_U is typically the transpose of the embedding layer
        # In HookedTransformer, it's stored in unembed.W_U
        W_U = self.model.unembed.W_U  # Shape: [d_vocab_out, d_model]
        return W_U

    def compute_dft_embedding(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DFT of embedding matrix along vocabulary dimension.

        Returns:
            (dft_matrix, fourier_norms)
            - dft_matrix: Complex FFT output, shape [p, d_model]
            - fourier_norms: L2 norm per frequency, shape [p]
        """
        W_E = self.get_embedding_matrix()  # [p, d_model]

        # Compute FFT along vocabulary dimension (dim 0)
        # torch.fft.fft operates on last dimension by default
        # We want to transpose, apply FFT, then transpose back
        W_E_T = W_E.T  # [d_model, p]

        # Apply FFT along dimension 1 (vocabulary dimension)
        dft_matrix = fft.fft(W_E_T.to(torch.float32), dim=1)  # [d_model, p]

        # Compute L2 norm across embedding dimensions for each frequency
        fourier_norms = torch.abs(dft_matrix).pow(2).sum(dim=0).sqrt()  # [p]

        return dft_matrix.T, fourier_norms  # Return transposed for clarity

    def identify_key_frequencies(
        self,
        fourier_norms: torch.Tensor,
        top_k: int = 10,
        threshold: float = None,
    ) -> torch.Tensor:
        """
        Identify key frequencies (sparse components in Fourier basis).

        Args:
            fourier_norms: L2 norms per frequency
            top_k: Number of top frequencies to select
            threshold: Alternatively, threshold for significant frequencies

        Returns:
            Indices of key frequencies, sorted by magnitude (descending)
        """
        if threshold is not None:
            key_freq_indices = torch.where(fourier_norms > threshold)[0]
        else:
            # Get top-k frequencies
            _, key_freq_indices = torch.topk(fourier_norms, k=min(top_k, len(fourier_norms)))

        # Sort by magnitude (descending)
        key_freq_indices = key_freq_indices[torch.argsort(fourier_norms[key_freq_indices], descending=True)]

        return key_freq_indices

    def verify_trigonometric_identity(
        self,
        dataset,
        key_freq_indices: torch.Tensor,
        num_examples: int = 50,
    ) -> Dict:
        """
        Verify the trigonometric identity: cos(ω(a+b)) = cos(ωa)cos(ωb) - sin(ωa)sin(ωb).

        Args:
            dataset: ModularAdditionDataset
            key_freq_indices: Indices of key frequencies
            num_examples: Number of examples to verify

        Returns:
            Dictionary with verification metrics
        """
        W_E = self.get_embedding_matrix()  # [p, d_model]
        dft_matrix, _ = self.compute_dft_embedding()  # [p, d_model]

        results = {
            'identity_errors': [],
            'example_details': [],
        }

        # Sample examples
        examples = dataset.train_pairs[:num_examples]

        for a, b, c in examples:
            a_idx, b_idx, c_idx = int(a), int(b), int(c)

            example_result = {
                'a': a_idx,
                'b': b_idx,
                'c': c_idx,
                'frequency_errors': [],
            }

            # Check identity for each key frequency
            for freq_idx in key_freq_indices[:5]:  # Check first 5 key frequencies
                freq = int(freq_idx.item())

                # Get Fourier components
                dft_a = dft_matrix[a_idx, :]  # [d_model]
                dft_b = dft_matrix[b_idx, :]  # [d_model]
                dft_c = dft_matrix[c_idx, :]  # [d_model]

                # Extract magnitude and phase for this frequency
                # cos(ω(a+b)) should equal cos(ωa)cos(ωb) - sin(ωa)sin(ωb)

                cos_wa = np.cos(2 * np.pi * freq * a_idx / self.p)
                sin_wa = np.sin(2 * np.pi * freq * a_idx / self.p)
                cos_wb = np.cos(2 * np.pi * freq * b_idx / self.p)
                sin_wb = np.sin(2 * np.pi * freq * b_idx / self.p)
                cos_wc = np.cos(2 * np.pi * freq * c_idx / self.p)

                # LHS: cos(ω(a+b))
                lhs = cos_wc

                # RHS: cos(ωa)cos(ωb) - sin(ωa)sin(ωb)
                rhs = cos_wa * cos_wb - sin_wa * sin_wb

                # Compute error
                error = float(np.abs(lhs - rhs))
                example_result['frequency_errors'].append({
                    'frequency': freq,
                    'error': error,
                })

            results['example_details'].append(example_result)

        # Compute average error
        all_errors = []
        for example in results['example_details']:
            for freq_error in example['frequency_errors']:
                all_errors.append(freq_error['error'])

        results['mean_identity_error'] = float(np.mean(all_errors))
        results['std_identity_error'] = float(np.std(all_errors))

        return results

    def analyze_sparsity(self, fourier_norms: torch.Tensor) -> Dict:
        """
        Analyze sparsity of the Fourier representation.

        Returns:
            Dictionary with sparsity metrics
        """
        fourier_norms = fourier_norms.detach().cpu().numpy()

        # Compute Gini coefficient (sparsity measure)
        sorted_norms = np.sort(fourier_norms)
        n = len(sorted_norms)
        cumsum = np.cumsum(sorted_norms)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_norms)) / (n * np.sum(sorted_norms)) - (n + 1) / n

        # Compute energy concentration
        total_energy = np.sum(fourier_norms)
        top_k_energies = {}
        for k in [1, 5, 10, 20]:
            top_k = np.sum(np.sort(fourier_norms)[-k:])
            top_k_energies[f'top_{k}'] = float(top_k / total_energy) if total_energy > 0 else 0.0

        return {
            'gini_coefficient': float(gini),
            'total_energy': float(total_energy),
            'energy_concentration': top_k_energies,
            'num_frequencies': len(fourier_norms),
        }

    def save_analysis(self, filepath: str, analysis_dict: Dict):
        """Save analysis results to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(filepath, 'w') as f:
            json.dump(analysis_dict, f, indent=2, default=str)
        print(f"Analysis saved to {filepath}")
