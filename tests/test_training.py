import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from facial_recognition_system.data import ImagePair
from facial_recognition_system.experiments import (
    AugmentationConfig,
    EmbeddingSet,
    pair_paths,
    pair_scores,
    read_pairs,
)
from facial_recognition_system.models import FaceEmbedder
from facial_recognition_system.preprocessing import Normalization
from facial_recognition_system.training import (
    ArcFaceHead,
    FaceDataset,
    build_augmentation,
    fit,
    freeze_except,
    label_indices,
    select_training_subset,
    set_train_mode,
)
from scripts.train import main

TRAINABLE = ["last_linear", "last_bn"]


class TinyNetwork(nn.Module):
    """Same layout as the end of InceptionResnetV1: frozen body, dropout, linear and batch norm."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=8, stride=8),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(4, 8, bias=False)
        self.last_bn = nn.BatchNorm1d(8)

    def forward(self, faces: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.body(faces).flatten(1))
        return self.last_bn(self.last_linear(features))


class TinyEmbedder(FaceEmbedder):
    model_id = "tiny"
    embedding_dim = 8
    normalization = Normalization.FACENET

    def __init__(self) -> None:
        super().__init__()
        self.network = TinyNetwork()

    def extract(self, faces: torch.Tensor) -> torch.Tensor:
        return self.network(faces)


def random_batches(count: int, batch_size: int = 4, classes: int = 3):
    generator = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(batch_size, 3, 32, 32, generator=generator),
            torch.randint(0, classes, (batch_size,), generator=generator),
        )
        for _ in range(count)
    ]


def make_fit_parts(tmp_path: Path):
    torch.manual_seed(0)
    embedder = TinyEmbedder()
    head = ArcFaceHead(embedding_dim=8, num_classes=3)
    parameters = freeze_except(embedder.network, TRAINABLE)
    optimizer = torch.optim.AdamW([*parameters, *head.parameters()], lr=1e-2)
    return embedder, head, optimizer, tmp_path / "checkpoints" / "best.pt"


# ArcFace head


@pytest.mark.unit
def test_arcface_logits_have_one_column_per_identity_and_are_bounded_by_scale() -> None:
    head = ArcFaceHead(embedding_dim=8, num_classes=5, scale=32.0, margin=0.5)

    logits = head(torch.randn(6, 8), torch.tensor([0, 1, 2, 3, 4, 0]))

    assert logits.shape == (6, 5)
    assert logits.abs().max() <= 32.0 + 1e-4


@pytest.mark.unit
def test_arcface_margin_lowers_only_the_true_class_logit() -> None:
    with_margin = ArcFaceHead(embedding_dim=8, num_classes=4, margin=0.5)
    without_margin = ArcFaceHead(embedding_dim=8, num_classes=4, margin=0.0)
    without_margin.load_state_dict(with_margin.state_dict())
    embeddings = torch.randn(10, 8)
    labels = torch.randint(0, 4, (10,))

    penalized = with_margin(embeddings, labels)
    plain = without_margin(embeddings, labels)

    true_class = torch.zeros_like(plain, dtype=torch.bool)
    true_class[torch.arange(10), labels] = True
    assert torch.allclose(penalized[~true_class], plain[~true_class])
    assert (penalized[true_class] < plain[true_class]).all()


@pytest.mark.unit
def test_arcface_without_margin_is_scaled_cosine_similarity() -> None:
    head = ArcFaceHead(embedding_dim=8, num_classes=3, scale=10.0, margin=0.0)
    embeddings = torch.randn(4, 8)

    logits = head(embeddings, torch.tensor([0, 1, 2, 0]))

    cosine = nn.functional.normalize(embeddings) @ nn.functional.normalize(head.weight).T
    assert torch.allclose(logits, 10.0 * cosine, atol=1e-5)


# Partial freezing


@pytest.mark.unit
def test_freeze_except_leaves_only_the_trainable_children_with_gradients() -> None:
    network = TinyNetwork()

    parameters = freeze_except(network, TRAINABLE)

    trainable_names = {
        name for name, parameter in network.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {"last_linear.weight", "last_bn.weight", "last_bn.bias"}
    assert len(parameters) == 3


@pytest.mark.unit
def test_freeze_except_rejects_unknown_children() -> None:
    with pytest.raises(ValueError, match="block8"):
        freeze_except(TinyNetwork(), ["block8"])


@pytest.mark.unit
def test_set_train_mode_keeps_frozen_batch_norms_in_eval_mode() -> None:
    network = TinyNetwork()

    set_train_mode(network, TRAINABLE)

    assert not network.body[1].training
    assert network.last_bn.training
    assert network.dropout.training


@pytest.mark.unit
def test_frozen_batch_norm_statistics_do_not_change_during_training() -> None:
    network = TinyNetwork()
    frozen_mean = network.body[1].running_mean.clone()
    trainable_mean = network.last_bn.running_mean.clone()

    set_train_mode(network, TRAINABLE)
    network(torch.randn(4, 3, 32, 32) + 5)

    assert torch.equal(network.body[1].running_mean, frozen_mean)
    assert not torch.equal(network.last_bn.running_mean, trainable_mean)


# Fine-tuning subset and dataset


@pytest.mark.unit
def test_select_training_subset_drops_small_identities_and_caps_large_ones() -> None:
    paths = (
        [f"Ada/Ada_{i:04d}.png" for i in range(3)]
        + [f"Alan/Alan_{i:04d}.png" for i in range(4)]
        + [f"Bush/Bush_{i:04d}.png" for i in range(10)]
    )

    subset = select_training_subset(paths, min_photos=4, max_photos=6, seed=42)

    assert not any(path.startswith("Ada/") for path in subset)
    assert sum(path.startswith("Alan/") for path in subset) == 4
    assert sum(path.startswith("Bush/") for path in subset) == 6
    assert subset == select_training_subset(list(reversed(paths)), 4, 6, seed=42)
    assert subset != select_training_subset(paths, 4, 6, seed=7)


@pytest.mark.unit
def test_label_indices_are_contiguous_and_sorted_by_identity() -> None:
    labels = label_indices(["Bob/Bob_1.png", "Ada/Ada_1.png", "Bob/Bob_2.png", "Ada/Ada_2.png"])

    assert labels == {"Ada": 0, "Bob": 1}


@pytest.mark.unit
def test_build_augmentation_returns_none_when_everything_is_off() -> None:
    config = AugmentationConfig(horizontal_flip=False, brightness=0, contrast=0, saturation=0)

    assert build_augmentation(config) is None
    assert build_augmentation(AugmentationConfig()) is not None


@pytest.mark.unit
def test_face_dataset_without_augmentation_returns_the_same_tensor_every_time(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (160, 160), (10, 20, 30))
    image.paste((200, 100, 50), (0, 0, 80, 160))
    (tmp_path / "Ada").mkdir()
    image.save(tmp_path / "Ada" / "Ada_1.png")
    plain = FaceDataset(tmp_path, ["Ada/Ada_1.png"], {"Ada": 0}, TinyEmbedder().preprocess)
    augmented = FaceDataset(
        tmp_path,
        ["Ada/Ada_1.png"],
        {"Ada": 0},
        TinyEmbedder().preprocess,
        build_augmentation(AugmentationConfig(horizontal_flip=True, brightness=0.5)),
    )

    torch.manual_seed(0)
    first, label = plain[0]
    assert label == 0
    assert torch.equal(first, plain[0][0])
    assert any(not torch.equal(first, augmented[0][0]) for _ in range(10))


# Training loop


@pytest.mark.unit
def test_fit_stops_after_patience_epochs_without_improvement(tmp_path: Path) -> None:
    embedder, head, optimizer, checkpoint_path = make_fit_parts(tmp_path)
    eers = iter([0.10, 0.08, 0.05, 0.06, 0.07, 0.01])
    saved_states = []

    def validate() -> float:
        saved_states.append(copy.deepcopy(embedder.network.state_dict()))
        return next(eers)

    history = fit(
        embedder,
        head,
        random_batches(3),
        validate,
        optimizer,
        trainable=TRAINABLE,
        max_epochs=10,
        patience=2,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert [record.epoch for record in history.epochs] == [0, 1, 2, 3, 4]
    assert history.initial_val_eer == 0.10
    assert history.best_epoch == 2
    assert history.best_val_eer == 0.05
    assert history.stopped_early
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["epoch"] == 2
    for name, value in saved_states[2].items():
        assert torch.equal(checkpoint["network"][name], value)


@pytest.mark.unit
def test_fit_keeps_a_trained_epoch_even_when_the_pretrained_model_was_better(
    tmp_path: Path,
) -> None:
    embedder, head, optimizer, checkpoint_path = make_fit_parts(tmp_path)
    eers = iter([0.01, 0.05, 0.04])

    history = fit(
        embedder,
        head,
        random_batches(2),
        lambda: next(eers),
        optimizer,
        trainable=TRAINABLE,
        max_epochs=2,
        patience=5,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert history.best_epoch == 2
    assert history.best_val_eer == 0.04
    assert not history.stopped_early


@pytest.mark.unit
def test_fit_trains_only_the_trainable_layers_and_respects_max_steps(tmp_path: Path) -> None:
    embedder, head, optimizer, checkpoint_path = make_fit_parts(tmp_path)
    before = copy.deepcopy(embedder.network.state_dict())
    batches = random_batches(5)
    seen = []

    def counting_batches():
        for batch in batches:
            seen.append(batch)
            yield batch

    class Batches:
        def __iter__(self):
            return counting_batches()

    history = fit(
        embedder,
        head,
        Batches(),
        lambda: 0.1,
        optimizer,
        trainable=TRAINABLE,
        max_epochs=1,
        patience=1,
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
        max_steps=2,
    )

    assert history.epochs[1].train_loss is not None
    assert len(seen) == 2
    after = embedder.network.state_dict()
    assert torch.equal(after["body.0.weight"], before["body.0.weight"])
    assert torch.equal(after["body.1.running_mean"], before["body.1.running_mean"])
    assert not torch.equal(after["last_linear.weight"], before["last_linear.weight"])


# Verification pairs


@pytest.mark.unit
def test_read_pairs_and_pair_scores(tmp_path: Path) -> None:
    pairs_file = tmp_path / "pairs.csv"
    pairs_file.write_text("path_a,path_b,same\nA/1.png,A/2.png,1\nA/1.png,B/1.png,0\n")
    embedding_set = EmbeddingSet(
        embeddings=torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]).numpy(),
        paths=["A/1.png", "A/2.png", "B/1.png"],
        labels=["A", "A", "B"],
        model_id="fake",
    )

    pairs = read_pairs(pairs_file)
    scores = pair_scores(embedding_set, pairs)

    assert pairs == [
        ImagePair(path_a="A/1.png", path_b="A/2.png", same=True),
        ImagePair(path_a="A/1.png", path_b="B/1.png", same=False),
    ]
    assert pair_paths(pairs) == ["A/1.png", "A/2.png", "B/1.png"]
    assert scores.scores.tolist() == pytest.approx([0.6, 0.0])
    assert scores.labels.tolist() == [1, 0]


# Script


@pytest.fixture
def training_root(tmp_path: Path) -> Path:
    colors = {"Ada": (200, 40, 40), "Alan": (40, 200, 40), "Grace": (40, 40, 200)}
    crops_dir = tmp_path / "crops"
    train, val = [], []
    for identity, (red, green, blue) in colors.items():
        (crops_dir / identity).mkdir(parents=True)
        for index in range(6):
            path = f"{identity}/{identity}_{index:04d}.png"
            Image.new("RGB", (160, 160), (red + index, green, blue)).save(crops_dir / path)
            (train if index < 4 else val).append(path)
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    (splits_dir / "lfw_train.txt").write_text("".join(f"{path}\n" for path in train))
    (splits_dir / "lfw_val.txt").write_text("".join(f"{path}\n" for path in val))
    (splits_dir / "lfw_test.txt").write_text("".join(f"{path}\n" for path in val))
    (splits_dir / "lfw_val_pairs.csv").write_text(
        "path_a,path_b,same\n"
        "Ada/Ada_0004.png,Ada/Ada_0005.png,1\n"
        "Alan/Alan_0004.png,Alan/Alan_0005.png,1\n"
        "Ada/Ada_0004.png,Grace/Grace_0004.png,0\n"
        "Alan/Alan_0005.png,Grace/Grace_0005.png,0\n"
    )
    (tmp_path / "E99.yaml").write_text(
        "experiment: E99\n"
        "hypothesis: a rede minúscula aprende as cores\n"
        "strategy: fine-tuning parcial com ArcFace\n"
        "model:\n  backbone: facenet\n  pretrained: false\n"
        "dataset:\n"
        f"  crops_dir: {crops_dir.as_posix()}\n"
        f"  splits_dir: {splits_dir.as_posix()}\n"
        "  splits: [val, test]\n"
        "embedding:\n  batch_size: 4\n"
        "training:\n"
        "  trainable: [last_linear, last_bn]\n"
        "  batch_size: 4\n"
        "  max_epochs: 3\n"
        "  patience: 2\n",
        encoding="utf-8",
    )
    return tmp_path


def run_training(root: Path, *extra: str) -> int:
    return main(
        [
            "--config",
            str(root / "E99.yaml"),
            "--device",
            "cpu",
            "--checkpoints-dir",
            str(root / "checkpoints"),
            "--embeddings-dir",
            str(root / "embeddings"),
            "--results-dir",
            str(root / "results"),
            *extra,
        ],
        embedder=TinyEmbedder(),
    )


@pytest.mark.unit
def test_smoke_run_saves_a_checkpoint_and_writes_no_results(training_root: Path) -> None:
    assert run_training(training_root, "--smoke") == 0

    assert (training_root / "checkpoints" / "smoke" / "E99" / "best.pt").exists()
    assert not (training_root / "results").exists()
    assert not (training_root / "embeddings").exists()


@pytest.mark.unit
def test_full_run_writes_embeddings_and_training_summary(training_root: Path) -> None:
    assert run_training(training_root) == 0

    record = json.loads((training_root / "results" / "E99.json").read_text(encoding="utf-8"))
    training = record["training"]
    assert record["model"] == "tiny-arcface-E99"
    assert training["subset"] == {"identities": 3, "photos": 12}
    assert training["loss"] == "arcface"
    assert training["trainable_parameters"] == 4 * 8 + 8 + 8
    assert 1 <= training["best_epoch"] <= training["epochs_run"] <= 3
    assert training["history"][0]["epoch"] == 0
    assert training["checkpoint"].endswith("checkpoints/E99/best.pt")
    assert record["config"]["training"]["trainable"] == ["last_linear", "last_bn"]
    assert record["threshold"] is None
    assert (training_root / "embeddings" / "E99" / "test.npz").exists()


@pytest.mark.unit
def test_config_without_training_section_fails(training_root: Path, tmp_path: Path) -> None:
    config = training_root / "E99.yaml"
    config.write_text(config.read_text(encoding="utf-8").split("training:")[0], encoding="utf-8")

    assert run_training(training_root) == 1
