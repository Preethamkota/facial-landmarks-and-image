from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _plot_metric(ax, epochs, train_values, val_values, title, ylabel) -> None:
    ax.plot(epochs, train_values, label="Train", linewidth=2)
    ax.plot(epochs, val_values, label="Validation", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_confusion_matrix(ax, matrix, labels) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=color)


def create_training_report(
    output_dir: Path,
    labels: list[str],
    history: dict[str, list[float]],
    confusion_matrix: list[list[int]],
    summary: dict[str, float | int | str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"{timestamp}.pdf"

    epochs = list(range(1, len(history["train_loss"]) + 1))

    with PdfPages(report_path) as pdf:
        summary_fig = plt.figure(figsize=(8.27, 11.69))
        summary_fig.suptitle("Training Metrics Report", fontsize=18, y=0.97)
        summary_ax = summary_fig.add_subplot(111)
        summary_ax.axis("off")

        summary_lines = [
            f"Generated: {summary['generated_at']}",
            f"Epochs: {summary['epochs']}",
            f"Dataset size: {summary['dataset_size']}",
            f"Train samples: {summary['train_size']}",
            f"Validation samples: {summary['val_size']}",
            f"Final train loss: {summary['final_train_loss']:.4f}",
            f"Final train accuracy: {summary['final_train_acc']:.4f}",
            f"Final validation loss: {summary['final_val_loss']:.4f}",
            f"Final validation accuracy: {summary['final_val_acc']:.4f}",
            f"Best validation accuracy: {summary['best_val_acc']:.4f}",
            f"Best model path: {summary['best_model_path']}",
            f"Final model path: {summary['final_model_path']}",
            f"Labels: {', '.join(labels)}",
        ]
        summary_ax.text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            va="top",
            ha="left",
            fontsize=12,
            family="monospace",
        )
        pdf.savefig(summary_fig, bbox_inches="tight")
        plt.close(summary_fig)

        metric_fig, metric_axes = plt.subplots(2, 1, figsize=(11, 8.5))
        metric_fig.suptitle("Training Curves", fontsize=16)
        _plot_metric(metric_axes[0], epochs, history["train_loss"], history["val_loss"], "Loss", "Loss")
        _plot_metric(metric_axes[1], epochs, history["train_acc"], history["val_acc"], "Accuracy", "Accuracy")
        metric_fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(metric_fig, bbox_inches="tight")
        plt.close(metric_fig)

        cm_fig, cm_ax = plt.subplots(figsize=(8.5, 8))
        _plot_confusion_matrix(cm_ax, confusion_matrix, labels)
        cm_fig.tight_layout()
        pdf.savefig(cm_fig, bbox_inches="tight")
        plt.close(cm_fig)

    return report_path
