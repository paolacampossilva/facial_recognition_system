"""Training loop with early stopping on the validation EER (decision D5)."""

import time
from collections.abc import Callable, Iterable, Sequence
from itertools import islice
from pathlib import Path

import torch
from pydantic import BaseModel
from torch import nn
from torch.nn.functional import cross_entropy

from facial_recognition_system.models import FaceEmbedder
from facial_recognition_system.training.freezing import set_train_mode


class EpochRecord(BaseModel):
    """One epoch; epoch 0 is the pretrained model, validated before any training."""

    epoch: int
    train_loss: float | None
    val_eer: float
    seconds: float


class TrainingHistory(BaseModel):
    epochs: list[EpochRecord]
    best_epoch: int
    best_val_eer: float
    stopped_early: bool
    seconds: float

    @property
    def initial_val_eer(self) -> float:
        return self.epochs[0].val_eer


def fit(
    embedder: FaceEmbedder,
    head: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    validate: Callable[[], float],
    optimizer: torch.optim.Optimizer,
    *,
    trainable: Sequence[str],
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
    device: torch.device,
    max_steps: int | None = None,
) -> TrainingHistory:
    """Trains the embedder through the head and keeps the epoch with the lowest validation EER.

    The validation EER is measured once before training (epoch 0, the
    pretrained model) as the reference, and after every epoch. Only trained
    epochs compete for the best checkpoint, so the result is always a
    fine-tuned model, and the history shows whether it beat the reference.
    Training stops after ``patience`` epochs without improvement.

    Args:
        embedder: Wrapper whose ``network`` is fine-tuned, already on ``device``.
        head: Classification head, such as ``ArcFaceHead``, already on ``device``.
        batches: Training batches of (faces, class indices); iterated once per epoch.
        validate: Returns the validation EER of the embedder in its current state.
        optimizer: Optimizer over the trainable parameters and the head.
        trainable: Children of ``embedder.network`` that train (the rest stays frozen).
        max_epochs: Maximum number of epochs.
        patience: Epochs without a lower EER before stopping.
        checkpoint_path: Where the best epoch is saved (outside Git).
        device: Device of the embedder and head.
        max_steps: Stops each epoch after this many batches (``--smoke``).

    Returns:
        The EER and loss of every epoch and which one was kept.
    """
    start = time.perf_counter()
    embedder.eval()
    epochs = [EpochRecord(epoch=0, train_loss=None, val_eer=validate(), seconds=0.0)]
    best_epoch = 0
    best_val_eer = float("inf")
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        set_train_mode(embedder.network, trainable)
        head.train()
        losses = []
        for faces, labels in islice(batches, max_steps):
            labels = labels.to(device)
            loss = cross_entropy(head(embedder(faces.to(device)), labels), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        embedder.eval()
        val_eer = validate()
        epochs.append(
            EpochRecord(
                epoch=epoch,
                train_loss=sum(losses) / len(losses) if losses else None,
                val_eer=val_eer,
                seconds=round(time.perf_counter() - epoch_start, 2),
            )
        )

        if val_eer < best_val_eer:
            best_epoch, best_val_eer = epoch, val_eer
            save_checkpoint(checkpoint_path, embedder, head, epoch, val_eer)
        elif epoch - best_epoch >= patience:
            stopped_early = True
            break

    return TrainingHistory(
        epochs=epochs,
        best_epoch=best_epoch,
        best_val_eer=best_val_eer,
        stopped_early=stopped_early,
        seconds=round(time.perf_counter() - start, 2),
    )


def save_checkpoint(
    path: Path, embedder: FaceEmbedder, head: nn.Module, epoch: int, val_eer: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": embedder.model_id,
            "epoch": epoch,
            "val_eer": val_eer,
            "network": embedder.network.state_dict(),
            "head": head.state_dict(),
        },
        path,
    )


def load_checkpoint(path: Path, embedder: FaceEmbedder) -> int:
    """Loads the saved network weights into ``embedder`` and returns the epoch."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    embedder.network.load_state_dict(checkpoint["network"])
    return int(checkpoint["epoch"])
