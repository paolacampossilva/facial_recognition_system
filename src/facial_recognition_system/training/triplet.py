"""Batch-hard triplet loss (Hermans et al., 2017) with the FaceNet margin, decision D5."""

import torch
from torch import nn
from torch.nn.functional import normalize


def batch_hard_triplet_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    """Mean over the anchors of ``max(0, d(a, hardest p) - d(a, hardest n) + margin)``.

    For each photo (the anchor), the hardest positive is the farthest photo of
    the same person in the batch and the hardest negative is the closest photo
    of someone else. Easy triplets give zero loss and teach nothing, so only the
    hardest ones are used. The distance is the squared Euclidean distance
    between L2-normalized embeddings (``2 - 2 cos``), the one the FaceNet paper
    used with the same margin of 0.2.

    Anchors without a positive or a negative in the batch are skipped; the P x K
    sampler makes sure every anchor has both.
    """
    embeddings = normalize(embeddings, dim=1)
    distances = (2 - 2 * embeddings @ embeddings.T).clamp(min=0)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    itself = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~itself
    negatives = ~same

    hardest_positive = distances.masked_fill(~positives, float("-inf")).amax(dim=1)
    hardest_negative = distances.masked_fill(~negatives, float("inf")).amin(dim=1)
    valid = positives.any(dim=1) & negatives.any(dim=1)
    if not valid.any():
        # Keeps the graph connected so backward() still works on a degenerate batch.
        return embeddings.sum() * 0.0
    losses = (hardest_positive - hardest_negative + margin).clamp(min=0)
    return losses[valid].mean()


class TripletLoss(nn.Module):
    """``loss(embeddings, labels)``, like ``ArcFaceLoss``, with no parameters to train."""

    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return batch_hard_triplet_loss(embeddings, labels, self.margin)
