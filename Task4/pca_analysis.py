"""PCA analysis of embedding matrix to verify circular representation."""
import torch
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA


class EmbeddingPCAAnalysis:
    """Perform PCA on embedding matrix."""

    def __init__(self, model, p: int = 113, device: torch.device = None):
        """
        Initialize PCA analysis.

        Args:
            model: HookedTransformer model
            p: Modulus
            device: torch.device
        """
        self.model = model
        self.p = p
        self.device = device or torch.device("cpu")
        self.pca = None
        self.embeddings_2d = None

    def get_embedding_matrix(self) -> torch.Tensor:
        """Extract embedding matrix."""
        W_E = self.model.embed.W_E  # [vocab, d_model]
        return W_E.cpu().detach().numpy()

    def fit_pca(self, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit PCA on embedding matrix.

        Args:
            n_components: Number of principal components (typically 2 for visualization)

        Returns:
            (embeddings_2d, explained_variance_ratio)
        """
        W_E = self.get_embedding_matrix()  # [vocab, d_model]

        self.pca = PCA(n_components=n_components)
        embeddings_2d = self.pca.fit_transform(W_E)  # [vocab, n_components]

        self.embeddings_2d = embeddings_2d

        explained_var = self.pca.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_var}")
        print(f"Cumulative explained variance: {np.cumsum(explained_var)}")

        return embeddings_2d, explained_var

    def compute_circularity_metrics(self) -> dict:
        """
        Compute metrics to assess whether embeddings form a circle.

        Returns:
            Dictionary with circularity metrics
        """
        if self.embeddings_2d is None:
            raise ValueError("PCA not fitted yet")

        embeddings = self.embeddings_2d  # [vocab, 2]

        # Compute center
        center = embeddings.mean(axis=0)

        # Compute distances from center
        distances_from_center = np.linalg.norm(embeddings - center, axis=1)

        # Check circularity: if points form a circle, distances should be roughly constant
        mean_distance = np.mean(distances_from_center)
        std_distance = np.std(distances_from_center)
        cv_distance = std_distance / mean_distance if mean_distance > 0 else float('inf')

        # Compute angles for each point
        centered_embeddings = embeddings - center
        angles = np.arctan2(centered_embeddings[:, 1], centered_embeddings[:, 0])
        angles = np.sort(angles)

        # Check if angles are evenly distributed
        angle_diffs = np.diff(np.concatenate([[angles[-1] - 2*np.pi], angles]))
        mean_angle_diff = np.mean(angle_diffs)
        std_angle_diff = np.std(angle_diffs)
        cv_angle = std_angle_diff / mean_angle_diff if mean_angle_diff > 0 else float('inf')

        results = {
            'mean_distance_from_center': float(mean_distance),
            'std_distance_from_center': float(std_distance),
            'cv_distance': float(cv_distance),  # Lower is better (more circular)
            'mean_angle_spacing': float(mean_angle_diff),
            'std_angle_spacing': float(std_angle_diff),
            'cv_angle': float(cv_angle),  # Lower is better (more uniform)
            'is_circular': cv_distance < 0.2 and cv_angle < 0.3,  # Heuristic thresholds
        }

        return results

    def compute_phase_structure(self) -> np.ndarray:
        """
        Compute phase (angle) for each embedding on the circle.

        Returns:
            Array of angles (in radians) for each token
        """
        if self.embeddings_2d is None:
            raise ValueError("PCA not fitted yet")

        embeddings = self.embeddings_2d
        center = embeddings.mean(axis=0)
        centered_embeddings = embeddings - center

        phases = np.arctan2(centered_embeddings[:, 1], centered_embeddings[:, 0])

        return phases

    def analyze_phase_structure(self) -> dict:
        """
        Analyze relationship between phase and token value.

        If the model learned modular arithmetic via rotation,
        phases should correlate with token values (mod p).

        Returns:
            Dictionary with phase-token correlations
        """
        phases = self.compute_phase_structure()

        # Use only the first p phases (exclude '=' token at index p)
        phases_tokens = phases[:self.p]

        results = {
            'phases': phases_tokens.tolist(),
            'tokens': list(range(self.p)),
            'phase_token_correlation': float(np.corrcoef(phases_tokens, np.arange(self.p))[0, 1]),
        }

        # Check phase differences for consecutive tokens
        phase_diffs = np.diff(phases_tokens)
        # Adjust for wraparound
        phase_diffs = np.where(np.abs(phase_diffs) > np.pi, np.sign(phase_diffs) * (2*np.pi - np.abs(phase_diffs)), phase_diffs)

        results['mean_phase_increment'] = float(np.mean(phase_diffs))
        results['std_phase_increment'] = float(np.std(phase_diffs))
        results['expected_phase_increment'] = 2 * np.pi / self.p

        return results

    def get_pca_coordinates(self) -> np.ndarray:
        """Get 2D PCA coordinates of all tokens."""
        if self.embeddings_2d is None:
            raise ValueError("PCA not fitted yet")
        return self.embeddings_2d

    def analyze_addition_geometry(self) -> dict:
        """
        Analyze geometric properties of addition on the circle.

        For a + b = c (mod p), if the model uses rotation:
        - The angle for c should be approximately angle(a) + angle(b)

        Returns:
            Dictionary with geometric analysis
        """
        phases = self.compute_phase_structure()  # [p]

        results = {
            'addition_examples': [],
            'mean_angle_error': [],
        }

        # Sample some addition examples
        for a in range(0, self.p, max(1, self.p // 20)):  # Sample ~20 examples
            for b in range(0, self.p, max(1, self.p // 20)):
                c = (a + b) % self.p

                # Expected phase for c based on rotation hypothesis
                expected_phase = (phases[a] + phases[b]) % (2 * np.pi)
                actual_phase = phases[c]

                # Compute angle error
                phase_error = np.abs(expected_phase - actual_phase)
                phase_error = min(phase_error, 2 * np.pi - phase_error)  # Shortest distance on circle

                results['addition_examples'].append({
                    'a': int(a),
                    'b': int(b),
                    'c': int(c),
                    'expected_phase': float(expected_phase),
                    'actual_phase': float(actual_phase),
                    'phase_error': float(phase_error),
                })

                results['mean_angle_error'].append(phase_error)

        if results['mean_angle_error']:
            results['mean_angle_error'] = float(np.mean(results['mean_angle_error']))
            results['std_angle_error'] = float(np.std(results['mean_angle_error']))
        else:
            results['mean_angle_error'] = None
            results['std_angle_error'] = None

        return results
