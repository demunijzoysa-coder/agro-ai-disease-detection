import argparse
from pathlib import Path
import json
from datetime import datetime
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

from config import DATA_DIR, REPORTS_DIR
from data import load_datasets


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to .keras model")
    args = parser.parse_args()

    _, _, test_ds = load_datasets(DATA_DIR)

    model = tf.keras.models.load_model(args.model)

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = model.predict(test_ds).ravel()

    # 0 = blast, 1 = healthy (sigmoid gives P(healthy))
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=["blast", "healthy"],
        output_dict=True
    )

    # ------------------------
    # Save confusion matrix
    # ------------------------
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()

    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # ------------------------
    # Save metrics
    # ------------------------
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ------------------------
    # Append run log (AUTO-CREATES FILE)
    # ------------------------
    run_log_path = REPORTS_DIR / "run_log.csv"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(Path(args.model).resolve()),
        "accuracy": report["accuracy"],
        "blast_precision": report["blast"]["precision"],
        "blast_recall": report["blast"]["recall"],
        "healthy_precision": report["healthy"]["precision"],
        "healthy_recall": report["healthy"]["recall"],
    }

    write_header = not run_log_path.exists()
    with open(run_log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # ------------------------
    # Done
    # ------------------------
    print("✅ Evaluation complete")
    print(f"- Confusion matrix saved to {cm_path}")
    print(f"- Metrics saved to {metrics_path}")
    print(f"- Run log updated at {run_log_path}")


if __name__ == "__main__":
    main()
