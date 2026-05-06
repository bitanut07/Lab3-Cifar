from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, Type, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

from classifier import (
    LogisticRegressionClassifier,
    NeuralNetworkClassifier,
    MNISTClassifier,
    CIFARMLPClassifier,
    CIFARCNNClassifier,
    CIFARCNNAugClassifier,
    CIFARCNNGAPClassifier,
)

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"


class ModelSpec(TypedDict):
    classifier: Type[MNISTClassifier]
    dataset: str


MODEL_TYPES: Dict[str, ModelSpec] = {
    "logistic": {"classifier": LogisticRegressionClassifier, "dataset": "mnist"},
    "nn": {"classifier": NeuralNetworkClassifier, "dataset": "mnist"},
    "cifar_mlp": {"classifier": CIFARMLPClassifier, "dataset": "cifar10"},
    "cnn": {"classifier": CIFARCNNClassifier, "dataset": "cifar10"},
    "cnn_aug": {"classifier": CIFARCNNAugClassifier, "dataset": "cifar10"},
    "cnn_gap": {"classifier": CIFARCNNGAPClassifier, "dataset": "cifar10"},
}


def _model_path(model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"{model_type}_model.keras")


def _data_path(dataset: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset}.npz")


def _set_seed(seed: int = 42) -> None:
    """Thiết lập seed để kết quả dễ tái lập."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _top_confused_pairs(cm: np.ndarray, top_k: int = 3) -> list[dict]:
    matrix = cm.copy()
    np.fill_diagonal(matrix, 0)
    labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                pairs.append((i, j, int(matrix[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [
        {"true": labels[i], "pred": labels[j], "count": c}
        for i, j, c in pairs[:top_k]
    ]


def _plot_curves(history: dict, model_type: str) -> str | None:
    if not history:
        return None
    if not {"loss", "accuracy", "val_loss", "val_accuracy"}.issubset(history.keys()):
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{model_type}_curves.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_title(f"{model_type} loss")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train_acc")
    axes[1].plot(history["val_accuracy"], label="val_acc")
    axes[1].set_title(f"{model_type} accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_confusion(cm: np.ndarray, model_type: str) -> str:
    labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{model_type}_confusion.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_type} confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _configure_mnist() -> None:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Flatten 28x28 images to 784-dim vectors and normalise to [0, 1]
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

    np.savez(
        _data_path("mnist"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('mnist')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


def _configure_cifar10() -> None:
    local_npz_path = _data_path("cifar10")
    if os.path.exists(local_npz_path):
        print(f"Using local dataset at {local_npz_path} (skip download).")
        return

    local_7z_path = os.path.join(DATA_DIR, "cifar10.7z")
    if os.path.exists(local_7z_path):
        print("Found local data/cifar10.7z but data/cifar10.npz is missing.")
        print("Please extract data/cifar10.npz from the 7z file, then run configure again.")
        sys.exit(1)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Keep image tensors for CNN and normalise pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    np.savez(
        _data_path("cifar10"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )
    print(f"Dataset saved to {_data_path('cifar10')}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")


# ── configure ────────────────────────────────────────────────────────────────

def configure(args: argparse.Namespace) -> None:
    """Download and pre-process datasets used by the selected models."""

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.dataset in {"mnist", "all"}:
        _configure_mnist()
    if args.dataset in {"cifar10", "all"}:
        _configure_cifar10()


# ── train ────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """Train the selected model and save it to disk."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    _set_seed(42)
    if model_type == "cifar_mlp" and args.mlp_device == "cpu":
        # Ép MLP chạy CPU để tránh sai khác số học trên backend Metal.
        tf.config.set_visible_devices([], "GPU")

    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.model = classifier.build_model()

    print(f"Training {model_type} model on {spec['dataset']} …")
    callbacks: list[tf.keras.callbacks.Callback] = []
    if model_type == "cifar_mlp":
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=6,
                restore_best_weights=True,
            ),
        ]
    if model_type in {"cnn_aug", "cnn_gap"}:
        # Callback này hỗ trợ Task 5: giảm overfitting và giữ trọng số tốt nhất.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
            )
        ]
    history = classifier.train(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        callbacks=callbacks,
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = _model_path(model_type)
    classifier.save(save_path)
    print(f"Model saved to {save_path}")
    history_path = os.path.join(RESULTS_DIR, f"{model_type}_history.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)
    print(f"History saved to {history_path}")


# ── test ─────────────────────────────────────────────────────────────────────

def test(args: argparse.Namespace) -> None:
    """Evaluate the selected model and write results to a Markdown file."""
    model_type: str = args.model
    spec = MODEL_TYPES[model_type]
    data_path = _data_path(spec["dataset"])
    if not os.path.exists(data_path):
        sys.exit(
            f"Error: Dataset not found at {data_path}. "
            f"Run 'configure --dataset {spec['dataset']}' first."
        )

    model_path = _model_path(model_type)
    if not os.path.exists(model_path):
        sys.exit(f"Error: Trained model not found at {model_path}. Run 'train --model {model_type}' first.")

    data = np.load(data_path)
    x_test, y_test = data["x_test"], data["y_test"]

    cls = spec["classifier"]
    classifier = cls()
    classifier.load(model_path)

    print(f"Evaluating {model_type} model on {spec['dataset']} …")
    results = classifier.evaluate(x_test, y_test)

    y_pred = results["y_pred"]
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    top_pairs = _top_confused_pairs(cm, top_k=3)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    num_weights = classifier.model.count_params()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Markdown report
    md_path = os.path.join(RESULTS_DIR, f"{model_type}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {model_type.upper()} Model — Test Results\n\n")
        f.write(f"- **Dataset:** {spec['dataset']}\n")
        f.write(f"- **Loss:** {results['loss']:.4f}\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {precision:.4f}\n")
        f.write(f"- **Recall:** {recall:.4f}\n")
        f.write(f"- **F1-score:** {f1:.4f}\n")
        f.write(f"- **Weights:** {num_weights}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("```\n\n")
        f.write("## Confusion Matrix\n\n")
        f.write("```\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n```\n")
        f.write("\n## Top-3 Most Confused Class Pairs\n\n")
        for pair in top_pairs:
            f.write(f"- {pair['true']} -> {pair['pred']}: {pair['count']}\n")

    # JSON report
    json_path = os.path.join(RESULTS_DIR, f"{model_type}_results.json")
    json_data = {
        "model": model_type,
        "dataset": spec["dataset"],
        "weights": num_weights,
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "top_confused_pairs": top_pairs,
    }
    history_path = os.path.join(RESULTS_DIR, f"{model_type}_history.json")
    if os.path.exists(history_path):
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        json_data["history"] = history
        curves_path = _plot_curves(history, model_type)
        if curves_path:
            json_data["curves_figure"] = curves_path
            print(f"Curves figure saved to {curves_path}")
    conf_path = _plot_confusion(cm, model_type)
    json_data["confusion_figure"] = conf_path
    print(f"Confusion figure saved to {conf_path}")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Results saved to {md_path}")
    print(f"Results saved to {json_path}")


# ── summary ───────────────────────────────────────────────────────────────────

def summary(args: argparse.Namespace) -> None:
    """Read all JSON result files and write a combined summary.md."""
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "*_results.json")))
    if not json_files:
        sys.exit("Error: No result JSON files found. Run 'test' first.")

    records = []
    for path in json_files:
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))

    md_path = os.path.join(RESULTS_DIR, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Image Classification — Summary\n\n")
        f.write("| Model | Dataset | Weights | Accuracy | Precision | Recall | F1-score |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in records:
            f.write(
                f"| {r['model']} | {r.get('dataset', 'mnist')} | {r['weights']:,} "
                f"| {r['accuracy']:.4f} | {r['precision']:.4f} "
                f"| {r['recall']:.4f} | {r['f1_score']:.4f} |\n"
            )

    print(f"Summary saved to {md_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Image Classification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure
    configure_parser = subparsers.add_parser("configure", help="Download and prepare datasets")
    configure_parser.add_argument(
        "--dataset", default="mnist", choices=["mnist", "cifar10", "all"],
        help="Dataset to prepare (default: mnist)",
    )

    # train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to train (logistic | nn | cifar_mlp | cnn)",
    )
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    train_parser.add_argument("--validation-split", type=float, default=0.1, help="Validation split ratio")
    train_parser.add_argument(
        "--mlp-device",
        choices=["auto", "cpu", "gpu"],
        default="cpu",
        help="Device override for cifar_mlp training (default: cpu)",
    )

    # test
    test_parser = subparsers.add_parser("test", help="Evaluate a trained model")
    test_parser.add_argument(
        "--model", required=True, choices=MODEL_TYPES.keys(),
        help="Model type to evaluate (logistic | nn | cifar_mlp | cnn)",
    )

    # summary
    subparsers.add_parser("summary", help="Generate summary.md from all test results")

    args = parser.parse_args()

    commands = {
        "configure": configure,
        "train": train,
        "test": test,
        "summary": summary,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
