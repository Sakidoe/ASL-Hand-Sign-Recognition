# plot_asl_results.py
# In this helper script, we focus on visualizing the evaluation numbers.
# We take the CSV files produced by our extra_eval_asl.py script and turn them
# into clear plots that we can drop directly into our report or presentation.
# The idea is that we separate plotting from model code to keep things organized.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_class_accuracy(csv_path="per_class_accuracy.csv",
                            out_path="per_class_accuracy.png"):
    """
    Here we read the per_class_accuracy.csv file and draw a simple bar chart.
    Each bar corresponds to one letter, and the height is its test accuracy
    in percent. We save the figure as a PNG so we can reuse it later and also
    show it interactively so we can quickly inspect how balanced the model is.
    """
    df = pd.read_csv(csv_path)
    letters = df["letter"]
    acc = df["accuracy"] * 100.0

    plt.figure(figsize=(10, 5))
    plt.bar(letters, acc, color="skyblue")
    plt.ylim(0, 105)
    plt.xlabel("Letter")
    plt.ylabel("Accuracy (%)")
    plt.title("Per-class Accuracy on SignMNIST Test Set")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved per-class accuracy plot to {out_path}")
    plt.show()


def plot_confusion_matrix(csv_path="confusion_matrix.csv",
                          out_path="confusion_matrix.png"):
    """
    This function visualizes the confusion_matrix.csv as a heatmap.
    We treat each row as the true label index and each column as the
    predicted label index, and use a blue color map to show how often
    each pair occurs. The resulting figure makes it easy to spot which
    classes the model tends to confuse with each other.
    """
    cm = pd.read_csv(csv_path, header=0).values
    labels = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted label index")
    plt.ylabel("True label index")
    plt.title("Confusion Matrix (SignMNIST)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved confusion matrix plot to {out_path}")
    plt.show()


if __name__ == "__main__":
    """
    When we run this file directly, we generate both plots in one go.
    First we produce the per-class accuracy bar chart, then we plot
    the confusion matrix heatmap. This gives us two ready-made figures
    that we can paste into our final project report or slide deck.
    """
    plot_per_class_accuracy()
    plot_confusion_matrix()
