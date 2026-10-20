"""Partial fine-tuning (decision D5): only some top-level children of the backbone train."""

from collections.abc import Sequence

from torch import nn

BATCH_NORMS: tuple[type[nn.Module], ...] = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def freeze_except(network: nn.Module, trainable: Sequence[str]) -> list[nn.Parameter]:
    """Turns off the gradients of every parameter outside the ``trainable`` children.

    Args:
        network: Backbone network (``embedder.network``).
        trainable: Names of top-level children that keep training, such as ``block8``.

    Returns:
        The parameters that still train, for the optimizer.

    Raises:
        ValueError: A name in ``trainable`` is not a child of ``network``.
    """
    children = {name for name, _ in network.named_children()}
    unknown = sorted(set(trainable) - children)
    if unknown:
        raise ValueError(f"not children of the network: {', '.join(unknown)}")

    for name, parameter in network.named_parameters():
        parameter.requires_grad = name.split(".", 1)[0] in trainable
    return [parameter for parameter in network.parameters() if parameter.requires_grad]


def set_train_mode(network: nn.Module, trainable: Sequence[str]) -> None:
    """Puts the network in training mode, except the batch norms of the frozen part.

    Freezing the weights is not enough: a batch norm in training mode keeps
    updating its running mean and variance with every batch, so the "frozen"
    layers would still change. Those stay in eval mode. Dropout stays active,
    as in the original training, because it regularizes the trainable layers
    that follow it.
    """
    network.train()
    for name, child in network.named_children():
        if name in trainable:
            continue
        for module in child.modules():
            if isinstance(module, BATCH_NORMS):
                module.eval()
