"""Custom loss function with float64 precision to prevent slingshotting."""
import torch
import torch.nn.functional as F


def cross_entropy_loss_float64(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute cross-entropy loss with float64 precision.

    This custom implementation prevents "slingshotting" (sudden loss spikes
    from numerical overflow) by casting to float64 before softmax.

    Args:
        logits: Model logits, shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        targets: Target indices, shape [batch_size] or [batch_size, seq_len]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        Loss tensor (scalar or per-sample depending on reduction)
    """
    # Cast to float64 for numerical stability
    logits = logits.to(dtype=torch.float64)

    # Flatten if needed
    original_shape = logits.shape
    if logits.dim() == 3:
        # [batch, seq, vocab] -> [batch*seq, vocab]
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

    vocab_size = logits.shape[-1]

    # Compute log softmax
    log_softmax = F.log_softmax(logits, dim=-1)

    # Gather log probabilities for target classes
    # log_softmax shape: [batch, vocab]
    # targets shape: [batch]
    log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))
    log_probs = log_probs.squeeze(-1)

    # Compute negative log likelihood
    nll = -log_probs

    # Apply reduction
    if reduction == 'mean':
        loss = nll.mean()
    elif reduction == 'sum':
        loss = nll.sum()
    elif reduction == 'none':
        loss = nll
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Cast back to float32 for compatibility
    return loss.to(dtype=torch.float32)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: Model logits, shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        targets: Target indices, shape [batch_size] or [batch_size, seq_len]

    Returns:
        Accuracy as float in [0, 1]
    """
    # Flatten if needed
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = targets.reshape(-1)

    predictions = logits.argmax(dim=-1)
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()

    return accuracy
