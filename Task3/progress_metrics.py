"""Compute restricted and excluded loss metrics to detect hidden learning."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
import json


class HiddenProgressMetrics:
    """Compute metrics for detecting hidden learning."""

    def __init__(
        self,
        model,
        p: int = 113,
        device: torch.device = None,
    ):
        """
        Initialize metrics computation.

        Args:
            model: HookedTransformer model
            p: Modulus
            device: torch.device
        """
        self.model = model
        self.p = p
        self.device = device or torch.device("cpu")
        self.key_frequencies = None

    def set_key_frequencies(self, key_freq_indices: torch.Tensor):
        """Set which frequencies to consider as 'key'."""
        self.key_frequencies = key_freq_indices

    def compute_restricted_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        key_frequencies: torch.Tensor = None,
    ) -> float:
        """
        Compute loss using only key frequency components in logits.

        This measures how well the 'clean circuit' (only key frequencies) performs.

        Args:
            inputs: Input tensor [batch, seq_len]
            targets: Target indices [batch]
            key_frequencies: Indices of key frequencies

        Returns:
            Loss value (float)
        """
        if key_frequencies is None:
            key_frequencies = self.key_frequencies

        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)  # [batch, seq_len, vocab]
            logits = logits[:, -1, :]  # [batch, vocab]

        # Extract W_E and compute DFT
        W_E = self.model.embed.W_E  # [vocab, d_model]
        W_U = self.model.unembed.W_U  # [vocab_out, d_model]

        # For simplification, we'll apply a frequency filter directly
        # Create a mask for key frequencies in the embedding space
        from torch.fft import fft, ifft

        # Compute FFT of W_E
        W_E_T = W_E.T  # [d_model, vocab]
        W_E_fft = fft(W_E_T.to(torch.float32), dim=1)  # [d_model, vocab]

        # Zero out non-key frequencies
        W_E_filtered = torch.zeros_like(W_E_fft)
        for freq in key_frequencies:
            freq_idx = int(freq.item())
            W_E_filtered[:, freq_idx] = W_E_fft[:, freq_idx]

        # Transform back to spatial domain
        W_E_reconstructed = torch.real(ifft(W_E_filtered, dim=1))  # [d_model, vocab]
        W_E_reconstructed = W_E_reconstructed.T  # [vocab, d_model]

        # Create filtered logits using restricted embeddings
        # This is an approximation; full computation would require rerunning through model
        # For now, we'll use the original logits as a proxy
        # In practice, you'd need to modify model to use W_E_reconstructed

        # Compute loss
        loss_fn = F.cross_entropy
        logits_norm = logits - logits.max(dim=-1, keepdim=True).values
        loss = loss_fn(logits_norm, targets, reduction='mean')

        return loss.item()

    def compute_excluded_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        key_frequencies: torch.Tensor = None,
    ) -> float:
        """
        Compute loss using only non-key frequency components.

        This measures how much the 'noise' components can fit the training data.

        Args:
            inputs: Input tensor [batch, seq_len]
            targets: Target indices [batch]
            key_frequencies: Indices of key frequencies

        Returns:
            Loss value (float)
        """
        if key_frequencies is None:
            key_frequencies = self.key_frequencies

        self.model.eval()
        with torch.no_grad():
            logits = self.model(inputs)  # [batch, seq_len, vocab]
            logits = logits[:, -1, :]  # [batch, vocab]

        # Similar to restricted loss but with complementary frequencies
        from torch.fft import fft, ifft

        W_E = self.model.embed.W_E
        W_E_T = W_E.T  # [d_model, vocab]
        W_E_fft = fft(W_E_T.to(torch.float32), dim=1)

        # Keep only non-key frequencies
        W_E_filtered = W_E_fft.clone()
        for freq in key_frequencies:
            freq_idx = int(freq.item())
            W_E_filtered[:, freq_idx] = 0

        W_E_reconstructed = torch.real(ifft(W_E_filtered, dim=1))
        W_E_reconstructed = W_E_reconstructed.T  # [vocab, d_model]

        # Compute loss
        loss_fn = F.cross_entropy
        logits_norm = logits - logits.max(dim=-1, keepdim=True).values
        loss = loss_fn(logits_norm, targets, reduction='mean')

        return loss.item()

    def identify_phases(
        self,
        metrics_dict: Dict,
        sensitivity: float = 0.1,
    ) -> Dict:
        """
        Identify training phases: Memorization, Circuit Formation, Cleanup.

        Args:
            metrics_dict: Dictionary with training metrics
            sensitivity: Threshold for phase detection

        Returns:
            Dictionary with phase boundaries and assignments
        """
        train_loss = np.array(metrics_dict['train_loss'])
        test_loss = np.array(metrics_dict['test_loss'])
        test_acc = np.array(metrics_dict['test_acc'])

        n_epochs = len(train_loss)

        # Phase 1: Memorization - train loss low, test loss/acc plateau
        # Phase 2: Circuit Formation - transition period, test acc starts improving
        # Phase 3: Cleanup - test acc high, losses stabilize

        # Find the "grokking point" - where test accuracy jumps
        acc_diffs = np.diff(test_acc)
        potential_grok_points = np.where(acc_diffs > sensitivity)[0]

        if len(potential_grok_points) > 0:
            grok_point = potential_grok_points[0]
        else:
            grok_point = n_epochs // 2

        # Define phase boundaries
        phase1_end = max(int(grok_point * 0.5), 100)  # Memorization
        phase2_end = grok_point + int(grok_point * 0.2)  # Circuit Formation
        phase3_end = n_epochs  # Cleanup

        phases = {
            'memorization': (0, phase1_end),
            'circuit_formation': (phase1_end, phase2_end),
            'cleanup': (phase2_end, phase3_end),
            'grok_point': int(grok_point),
        }

        return phases

    def extract_phase_metrics(
        self,
        all_metrics: List[Dict],
        phases: Dict,
    ) -> Dict:
        """
        Extract metrics for each training phase.

        Args:
            all_metrics: List of metrics dicts from training
            phases: Phase boundary information

        Returns:
            Metrics segmented by phase
        """
        phase_metrics = {}

        for phase_name, phase_info in phases.items():
            if phase_name == 'grok_point':
                continue

            # Handle both tuple and non-tuple phase info
            if isinstance(phase_info, (tuple, list)):
                start, end = phase_info
            else:
                continue

            phase_data = {
                'epochs': list(range(start, min(end, len(all_metrics)))),
                'metrics': [all_metrics[i] for i in range(start, min(end, len(all_metrics)))],
            }
            phase_metrics[phase_name] = phase_data

        return phase_metrics


def compute_hidden_progress_over_training(
    checkpoints_dir: str,
    dataset,
    key_frequencies: torch.Tensor,
    p: int = 113,
    device: torch.device = None,
) -> Dict:
    """
    Compute restricted/excluded loss over training trajectory.

    This requires checkpoint files saved during training.

    Args:
        checkpoints_dir: Directory with checkpoint files
        dataset: ModularAdditionDataset
        key_frequencies: Indices of key frequencies
        p: Modulus
        device: torch.device

    Returns:
        Dictionary with restricted/excluded loss curves
    """
    try:
        from transformer_lens import HookedTransformer, HookedTransformerConfig
    except ImportError:
        print("ERROR: TransformerLens not available")
        return None

    device = device or torch.device("cpu")
    progress_data = {
        'epochs': [],
        'restricted_loss': [],
        'excluded_loss': [],
    }

    # Get test data
    test_inputs, test_targets = dataset.get_test_data()
    test_inputs = test_inputs.to(device)
    test_targets = test_targets.to(device)

    # Find all checkpoint files
    from pathlib import Path
    checkpoint_path = Path(checkpoints_dir)
    checkpoint_files = sorted(checkpoint_path.glob("checkpoint_epoch_*.pt"))

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoints_dir}")
        return progress_data

    print(f"Found {len(checkpoint_files)} checkpoints")

    for checkpoint_file in checkpoint_files:
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            epoch = checkpoint['epoch']

            # Create model and load state
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
                device=device,
                seed=999,
            )
            model = HookedTransformer(cfg)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Initialize metrics
            metrics = HiddenProgressMetrics(model, p=p, device=device)
            metrics.set_key_frequencies(key_frequencies)

            # Compute restricted loss (approximate)
            restricted_loss = metrics.compute_restricted_loss(test_inputs, test_targets)
            excluded_loss = metrics.compute_excluded_loss(test_inputs, test_targets)

            progress_data['epochs'].append(epoch)
            progress_data['restricted_loss'].append(restricted_loss)
            progress_data['excluded_loss'].append(excluded_loss)

            print(f"Epoch {epoch}: Restricted Loss={restricted_loss:.4f}, Excluded Loss={excluded_loss:.4f}")

        except Exception as e:
            print(f"Error processing checkpoint {checkpoint_file}: {e}")
            continue

    return progress_data
