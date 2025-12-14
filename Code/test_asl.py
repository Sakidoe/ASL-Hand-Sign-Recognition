# Evaluate the trained CNN on the Kaggle test CSV.

import numpy as np
import pandas as pd
import torch

from train_asl import SimpleCNN, index_to_letter, DEVICE


def load_test_csv(path: str):
    # load CSV where col0=label and remaining 784 cols are pixels
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values.astype(np.int64)
    pixels = df.iloc[:, 1:].values.astype(np.float32)
    pixels /= 255.0
    images = pixels.reshape(-1, 1, 28, 28)
    return torch.tensor(images, dtype=torch.float32).to(DEVICE), \
           torch.tensor(labels, dtype=torch.long).to(DEVICE)


def evaluate(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        acc = correct / labels.size(0)
    return acc, preds


def main():
    # load model
    model = SimpleCNN(num_classes=26)
    model.load_state_dict(torch.load("asl_cnn.pth", map_location=DEVICE))
    model.to(DEVICE)

    # load test set
    test_images, test_labels = load_test_csv("sign_mnist_test.csv")
    print("Loaded test set:", test_images.shape)

    # eval
    accuracy, preds = evaluate(model, test_images, test_labels)
    print(f"\nTest Accuracy on Kaggle test split: {accuracy * 100:.2f}%")

    # sample preds
    print("\nSample predictions (first 10):")
    for i in range(10):
        true_letter = index_to_letter(test_labels[i].item())
        pred_letter = index_to_letter(preds[i].item())
        print(f"Image {i}: True = {true_letter}  Pred = {pred_letter}")


if __name__ == "__main__":
    main()
