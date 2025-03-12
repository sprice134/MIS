#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def main():
    parser = argparse.ArgumentParser(
        description="Load evaluation metrics from CSV and plot ROC, Precision/Recall, F1, and Accuracy curves."
    )
    parser.add_argument("--csv", type=str, default="evaluation_metrics.csv",
                        help="Path to CSV file with evaluation metrics (must include columns: threshold, tn, fp, fn, tp, f1, accuracy).")
    parser.add_argument("--output_plot", type=str, default="metrics_plot.png",
                        help="Filename for the output plot image.")
    args = parser.parse_args()
    
    # Load CSV into a DataFrame.
    df = pd.read_csv(args.csv)
    
    # Ensure the required columns are available.
    required_columns = {"threshold", "tn", "fp", "fn", "tp", "f1", "accuracy"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_columns}")
    
    # Calculate FPR and TPR for ROC.
    df["fpr"] = df["fp"] / (df["fp"] + df["tn"])
    df["tpr"] = df["tp"] / (df["tp"] + df["fn"])
    
    # Compute precision and recall from confusion matrix components.
    def safe_div(n, d):
        return n / d if d != 0 else 0.0
    df["precision"] = df.apply(lambda row: safe_div(row["tp"], row["tp"] + row["fp"]), axis=1)
    df["recall"] = df.apply(lambda row: safe_div(row["tp"], row["tp"] + row["fn"]), axis=1)
    
    # Sort by threshold for plotting consistency.
    df_sorted = df.sort_values("threshold")
    
    # Compute the AUC of the ROC curve.
    roc_auc = auc(df_sorted["fpr"], df_sorted["tpr"])
    
    # Create a 2x2 grid of subplots.
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. ROC Curve
    ax = axs[0, 0]
    ax.plot(df_sorted["fpr"], df_sorted["tpr"], marker="o", label=f"ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(True)
    
    # 2. Precision and Recall vs. Threshold
    ax = axs[0, 1]
    ax.plot(df_sorted["threshold"], df_sorted["precision"], marker="o", label="Precision")
    ax.plot(df_sorted["threshold"], df_sorted["recall"], marker="o", label="Recall")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision and Recall vs. Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    ax.grid(True)
    
    # 3. F1 Score vs. Threshold
    ax = axs[1, 0]
    ax.plot(df_sorted["threshold"], df_sorted["f1"], marker="o", color="b")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score vs. Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # 4. Accuracy vs. Threshold
    ax = axs[1, 1]
    ax.plot(df_sorted["threshold"], df_sorted["accuracy"], marker="o", color="g")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Threshold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Metrics plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()

    '''
    python binaryEvalGraph.py \
        --csv binary_evaluation_metrics.csv \
        --output_plot temp_images/binary_roc_acc.png
    
    '''
