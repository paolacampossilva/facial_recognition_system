"""ArcFace classification head (additive angular margin, Deng et al., 2019), decision D5."""

import math

import torch
from torch import nn
from torch.nn.functional import linear, normalize


class ArcFaceHead(nn.Module):
    """Turns embeddings into logits over the training identities, with an angular margin.

    Each identity has a learned unit vector (its center). The logit of a class is
    ``scale * cos(theta)``, theta being the angle between the embedding and the
    center; for the true class, the angle is first increased by ``margin``. The
    model only lowers the loss by pulling each embedding at least ``margin``
    radians closer to its own center than to the others, which spreads the
    identities apart. The head is used only in training and discarded afterwards.
    """

    def __init__(
        self, embedding_dim: int, num_classes: int, scale: float = 32.0, margin: float = 0.5
    ) -> None:
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # Past theta = pi - margin, cos(theta + margin) would grow again; there the
        # penalty becomes linear instead (same fallback as the reference implementation).
        self._threshold = math.cos(math.pi - margin)
        self._linear_penalty = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns the ``(N, num_classes)`` logits for the cross-entropy loss."""
        cosine = linear(normalize(embeddings), normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        target = cosine.gather(1, labels.unsqueeze(1))
        sine = torch.sqrt(1 - target**2)
        with_margin = target * math.cos(self.margin) - sine * math.sin(self.margin)
        with_margin = torch.where(
            target > self._threshold, with_margin, target - self._linear_penalty
        )
        return self.scale * cosine.scatter(1, labels.unsqueeze(1), with_margin)
