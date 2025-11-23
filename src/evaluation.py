import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_sentiment(pred_path="outputs/sentiment/sentiment_predictions.csv"):
    print("\n========== EVALUATION: SENTIMENT ==========")

    df = pd.read_csv(pred_path)

    os.makedirs("outputs/evaluation", exist_ok=True)

    # Overall metrics
    y_true = df["true"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"[EVAL] Overall Accuracy = {acc:.3f}")
    print(f"[EVAL] Weighted Precision = {p:.3f}, Recall = {r:.3f}, F1 = {f1:.3f}")

    # Save metrics
    summary = pd.DataFrame(
        [["overall", acc, p, r, f1]],
        columns=["lang", "accuracy", "precision", "recall", "f1"]
    )
    summary.to_csv("outputs/evaluation/sentiment_overall.csv", index=False)

    # Per-language metrics
    lang_groups = []

    for lang in ["en", "es"]:
        df_lang = df[df["lang"] == lang]
        y_true_lang = df_lang["true"]
        y_pred_lang = df_lang["pred"]

        acc = accuracy_score(y_true_lang, y_pred_lang)
        p, r, f1, _ = precision_recall_fscore_support(y_true_lang, y_pred_lang, average="weighted")

        lang_groups.append([lang, acc, p, r, f1])
        print(f"[EVAL] {lang.upper()} F1 = {f1:.3f}")

    pd.DataFrame(lang_groups, columns=["lang", "accuracy", "precision", "recall", "f1"]) \
        .to_csv("outputs/evaluation/sentiment_by_lang.csv", index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "negative"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["positive", "negative"],
                yticklabels=["positive", "negative"])
    plt.title("Sentiment Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig("outputs/evaluation/confusion_matrix.png")
    plt.close()

    print("[EVAL] Confusion matrix saved.")

    return summary


def run_evaluation():
    print("\n========== FINAL EVALUATION ==========")
    evaluate_sentiment()
    print("[EVAL] Evaluation complete.")


if __name__ == "__main__":
    run_evaluation()
