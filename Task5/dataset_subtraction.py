"""Dataset for modular subtraction (non-commutative operation)."""
import torch
import numpy as np
from typing import Tuple


class ModularSubtractionDataset:
    """Create and manage modular subtraction dataset."""

    def __init__(self, p: int = 113, frac: float = 0.3, seed: int = 999):
        """
        Initialize dataset for a - b mod p.

        Args:
            p: Prime modulus
            frac: Fraction of data to use for training
            seed: Random seed
        """
        self.p = p
        self.frac = frac
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Generate all possible pairs
        self.all_pairs = []
        for a in range(p):
            for b in range(p):
                c = (a - b) % p
                self.all_pairs.append((a, b, c))

        # Shuffle and split
        np.random.shuffle(self.all_pairs)
        n_train = int(len(self.all_pairs) * frac)

        self.train_pairs = self.all_pairs[:n_train]
        self.test_pairs = self.all_pairs[n_train:]

        print(f"Subtraction Dataset created: P={p}, Train={len(self.train_pairs)}, Test={len(self.test_pairs)}")

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training data."""
        pairs = self.train_pairs
        return self._pairs_to_tensors(pairs)

    def get_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test data."""
        pairs = self.test_pairs
        return self._pairs_to_tensors(pairs)

    def _pairs_to_tensors(self, pairs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert (a, b, c) pairs to input/target tensors."""
        inputs = []
        targets = []

        for a, b, c in pairs:
            # Input: [a, b, -]  where - is encoded as p
            inp = torch.tensor([a, b, self.p], dtype=torch.long)
            targets.append(c)
            inputs.append(inp)

        inputs = torch.stack(inputs)
        targets = torch.tensor(targets, dtype=torch.long)

        return inputs, targets

    def get_full_train_loader(self, batch_size: int = 256):
        """Get training data loader."""
        inputs, targets = self.get_train_data()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_full_test_loader(self, batch_size: int = 256):
        """Get test data loader."""
        inputs, targets = self.get_test_data()
        dataset = torch.utils.data.TensorDataset(inputs, targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def __len__(self):
        return len(self.train_pairs) + len(self.test_pairs)
