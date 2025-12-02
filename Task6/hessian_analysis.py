"""Hessian eigenvalue analysis for loss landscape geometry."""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path


class HessianAnalysis:
    """Compute and analyze Hessian eigenvalues."""

    def __init__(self, model, device: torch.device = None):
        """
        Initialize Hessian analysis.

        Args:
            model: HookedTransformer model
            device: torch.device
        """
        self.model = model
        self.device = device or torch.device("cpu")

    def compute_loss_single_batch(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for a single batch."""
        logits = self.model(inputs)
        logits = logits[:, -1, :]  # [batch, vocab]

        # Use cross-entropy loss
        loss = F.cross_entropy(logits, targets, reduction='mean')
        return loss

    def compute_gradient_vector(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of loss w.r.t. model parameters.

        Returns:
            Flattened gradient vector
        """
        self.model.zero_grad()

        loss = self.compute_loss_single_batch(inputs, targets)
        loss.backward()

        # Collect gradients into single vector
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        grad_vector = torch.cat(grads)
        return grad_vector

    def compute_hessian_vector_product(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        vector: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute Hessian-vector product using finite differences.

        H*v ≈ (∇L(w + eps*v) - ∇L(w - eps*v)) / (2*eps)

        Args:
            inputs: Input tensor
            targets: Target tensor
            vector: Vector to multiply with Hessian
            eps: Epsilon for finite difference

        Returns:
            Hessian-vector product
        """
        # Save original parameters
        original_params = [p.clone() for p in self.model.parameters()]

        # Compute gradient at w
        grad_w = self.compute_gradient_vector(inputs, targets)

        # Compute gradient at w + eps*v
        self._perturb_parameters(vector, eps)
        grad_plus = self.compute_gradient_vector(inputs, targets)
        self._restore_parameters(original_params)

        # Compute gradient at w - eps*v
        self._perturb_parameters(vector, -eps)
        grad_minus = self.compute_gradient_vector(inputs, targets)
        self._restore_parameters(original_params)

        # Finite difference approximation
        hvp = (grad_plus - grad_minus) / (2 * eps)

        return hvp

    def _perturb_parameters(self, vector: torch.Tensor, scale: float):
        """Perturb model parameters by scale*vector."""
        offset = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data.add_(scale * vector[offset:offset + numel].view_as(param))
            offset += numel

    def _restore_parameters(self, original_params: List[torch.Tensor]):
        """Restore original parameters."""
        for param, original in zip(self.model.parameters(), original_params):
            param.data.copy_(original)

    def compute_top_eigenvalues_lanczos(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
        num_iterations: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top k eigenvalues of Hessian using Lanczos iteration.

        This is more efficient than full Hessian computation for large matrices.

        Args:
            inputs: Input tensor
            targets: Target tensor
            k: Number of eigenvalues to compute
            num_iterations: Number of Lanczos iterations

        Returns:
            (eigenvalues, eigenvectors) - both as numpy arrays
        """
        # Get parameter dimension
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params}")

        # Initialize random vector
        v = torch.randn(total_params, device=self.device)
        v = v / torch.norm(v)

        # Lanczos iteration
        V = [v]  # Orthonormal basis vectors
        T = []  # Tridiagonal matrix

        for j in range(num_iterations):
            # Compute Hessian-vector product
            w = self.compute_hessian_vector_product(inputs, targets, V[j])

            if j == 0:
                alpha = torch.dot(w, V[j])
                T.append([alpha.item()])
            else:
                alpha = torch.dot(w, V[j])
                w = w - alpha * V[j] - beta * V[j - 1]
                T[-1].append(beta.item())
                T.append([alpha.item()])

            # Gram-Schmidt orthogonalization
            for vi in V:
                w = w - torch.dot(w, vi) * vi

            beta = torch.norm(w)

            if beta < 1e-10 or j == num_iterations - 1:
                print(f"Lanczos converged at iteration {j}")
                break

            w = w / beta
            V.append(w)

        # Compute eigenvalues of tridiagonal matrix
        T_matrix = np.array(T)
        try:
            eigenvalues = np.linalg.eigvalsh(T_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            return eigenvalues[:k], None
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
            return np.array([]), None

    def compute_top_eigenvalues_power_method(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        k: int = 10,
        num_iterations: int = 50,
    ) -> np.ndarray:
        """
        Compute top k eigenvalues using power method + deflation.

        Args:
            inputs: Input tensor
            targets: Target tensor
            k: Number of eigenvalues
            num_iterations: Iterations per eigenvalue

        Returns:
            Array of top k eigenvalues
        """
        eigenvalues = []
        total_params = sum(p.numel() for p in self.model.parameters())

        for eig_idx in range(k):
            # Random initialization
            v = torch.randn(total_params, device=self.device)
            v = v / torch.norm(v)

            # Power iteration
            for _ in range(num_iterations):
                # Compute H*v
                Hv = self.compute_hessian_vector_product(inputs, targets, v)

                # Deflate by previously found eigenvectors
                for prev_v, prev_lambda in zip(eigenvalues, eigenvalues[:eig_idx]):
                    Hv = Hv - torch.dot(Hv, prev_v) * prev_v

                # Normalize
                lambda_k = torch.norm(Hv).item()
                if lambda_k > 1e-10:
                    v = Hv / lambda_k
                else:
                    break

            eigenvalues.append((v, lambda_k))
            print(f"Eigenvalue {eig_idx + 1}: {lambda_k:.6f}")

        # Extract eigenvalues
        eig_values = np.array([lam for _, lam in eigenvalues])
        eig_values = np.sort(eig_values)[::-1]  # Sort descending

        return eig_values

    def estimate_sharpness(self, eigenvalues: np.ndarray) -> Dict:
        """
        Estimate loss landscape sharpness from eigenvalues.

        Sharp minima have high eigenvalues.
        Flat minima have low eigenvalues.

        Returns:
            Dictionary with sharpness metrics
        """
        results = {
            'max_eigenvalue': float(np.max(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'mean_eigenvalue': float(np.mean(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'min_eigenvalue': float(np.min(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'eigenvalue_ratio': float(np.max(eigenvalues) / np.min(eigenvalues)) if len(eigenvalues) > 1 and np.min(eigenvalues) > 0 else float('inf'),
            'trace': float(np.sum(eigenvalues)),
            'condition_number': float(np.max(eigenvalues) / (np.min(eigenvalues) + 1e-8)),
        }

        return results


def compute_hessian_at_checkpoint(
    checkpoint_path: str,
    dataset,
    p: int = 113,
    device: torch.device = None,
) -> Dict:
    """
    Load checkpoint and compute Hessian eigenvalues.

    Args:
        checkpoint_path: Path to checkpoint file
        dataset: ModularAdditionDataset
        p: Modulus
        device: torch.device

    Returns:
        Dictionary with Hessian analysis results
    """
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    device = device or torch.device("cpu")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        epoch = checkpoint['epoch']

        # Create model
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

        # Get test data
        test_inputs, test_targets = dataset.get_test_data(batch_size=128)
        test_inputs = test_inputs[:128].to(device)  # Use subset for efficiency
        test_targets = test_targets[:128].to(device)

        # Compute Hessian
        hessian = HessianAnalysis(model, device=device)

        print(f"\nComputing Hessian for epoch {epoch}...")
        eigenvalues = hessian.compute_top_eigenvalues_power_method(
            test_inputs,
            test_targets,
            k=10,
            num_iterations=20
        )

        sharpness = hessian.estimate_sharpness(eigenvalues)

        results = {
            'epoch': epoch,
            'eigenvalues': eigenvalues.tolist(),
            'sharpness_metrics': sharpness,
        }

        return results

    except Exception as e:
        print(f"Error processing checkpoint {checkpoint_path}: {e}")
        return None
