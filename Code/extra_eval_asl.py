# extra_eval_asl.py
# In this extra script, we go beyond just reporting a single accuracy number.
# We evaluate our trained ASL CNN on the SignMNIST test split in a more detailed way.
# Specifically, we compute a confusion matrix and per-class accuracy so we can see
# which letters the model learns well and which letters are still confusing.

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from train_asl import SimpleCNN, index_to_letter, DEVICE


def load_test_csv(path: str):
    """
    Here we load the Kaggle SignMNIST test CSV and turn it into tensors.
    The first column is the numeric label and the remaining 784 columns are pixels.
    We normalize the pixels to [0,1] and reshape them to 1×28×28 so they match
    the input shape that our training pipeline and SimpleCNN expect.
    """
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values.astype(np.int64)
    pixels = df.iloc[:, 1:].values.astype(np.float32)
    pixels /= 255.0
    images = pixels.reshape(-1, 1, 28, 28)
    x = torch.tensor(images, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    return x, y


def main():
    """
    In main, we treat this file like a small evaluation experiment for our project.
    First we reload the trained CNN weights from disk and move the model to the
    same device we used during training (CPU or GPU). Then we load the official
    test split from Kaggle so we can evaluate our model under the same conditions.
    """
    # 1) Load model
    model = SimpleCNN(num_classes=26)
    state_dict = torch.load("asl_cnn.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    # 2) Load test set
    x_test, y_test = load_test_csv("sign_mnist_test.csv")
    print("Loaded test set:", x_test.shape)

    """
    Next we run the model in no-grad mode to avoid tracking gradients during testing.
    We compute a prediction for every test image in one pass and take the argmax
    over the logits to get the predicted class index. This gives us two aligned
    arrays: the true labels and the predicted labels for the whole test set.
    """
    # 3) Run inference
    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)

    y_true = y_test.cpu().numpy()
    y_pred = preds.cpu().numpy()

    """
    We start by computing the overall accuracy so we can compare with other models.
    This is the fraction of test examples where the predicted label matches the
    ground-truth label. Even though this is a single number, it is useful as a
    quick summary of how well our ASL classifier performs on SignMNIST.
    """
    # 4) Overall accuracy
    acc = (y_true == y_pred).mean()
    print(f"Overall test accuracy: {acc * 100:.2f}%")

    """
    To understand where the model makes mistakes, we build a confusion matrix.
    Each row corresponds to the true class index and each column to the predicted
    class index, so off-diagonal entries indicate misclassifications. We save
    the matrix as a CSV file so we can later plot it as a heatmap for our report.
    """
    # 5) Confusion matrix (raw label indices)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(26))
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv("confusion_matrix.csv", index=False)
    print("Saved confusion matrix to confusion_matrix.csv")

    """
    After that, we compute per-class accuracy to see which letters are easy or hard.
    For each label index that actually appears in the dataset, we filter the test
    samples belonging to that index and measure how often the model predicts them
    correctly. We also convert the numeric label to the human-readable letter.
    """
    # 6) Per-class accuracy for letters that actually exist in dataset
    letters = []
    class_acc = []
    for label_idx in range(26):
        mask = (y_true == label_idx)
        if mask.sum() == 0:
            continue  # labels like 9, 25 may be unused
        correct = (y_pred[mask] == label_idx).sum()
        acc_i = correct / mask.sum()
        letter = index_to_letter(label_idx)
        letters.append(letter)
        class_acc.append(acc_i)
        print(f"Letter {letter}: {acc_i * 100:.2f}% ({mask.sum()} samples)")

    """
    We store the per-class results in a small DataFrame so we can reuse them later.
    Saving this as per_class_accuracy.csv allows us to quickly load it in another
    script and draw bar charts or tables for our slides. This also keeps our
    evaluation pipeline modular and easy to extend for new models.
    """
    per_class_df = pd.DataFrame(
        {"letter": letters, "accuracy": class_acc}
    )
    per_class_df.to_csv("per_class_accuracy.csv", index=False)
    print("Saved per-class accuracy to per_class_accuracy.csv")

    """
    Finally, we sort the letters by accuracy and print the five hardest ones.
    This makes it very easy to talk about model limitations in our report because
    we can show exactly which letters the CNN struggles with the most. We can
    also use this list to guide future work, such as collecting more data for
    these specific signs or adding targeted data augmentation.
    """
    # 7) Show 5 hardest letters
    worst = per_class_df.sort_values("accuracy").head(5)
    print("\nFive hardest letters:")
    for _, row in worst.iterrows():
        print(f"{row['letter']}: {row['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
