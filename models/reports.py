import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_classification_report(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, report, cm


def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, ["No", "Yes"])
    plt.yticks(tick_marks, ["No", "Yes"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    # salvam imaginea
    if save_path is None:
        os.makedirs("outputs", exist_ok=True)
        safe_name = model_name.lower().replace(" ", "_")
        save_path = os.path.join("outputs", f"confusion_matrix_{safe_name}.png")

    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Confusion matrix salvată în: {save_path}")
