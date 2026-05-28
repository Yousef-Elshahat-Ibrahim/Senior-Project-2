import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_2xN_confusion_matrix(y_pred, y_true, encoder, model_name):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt

    # Resolves paths safely regardless of terminal execution context
    directory = Path.cwd() / "Results" / "Plots"
    directory.mkdir(parents=True, exist_ok=True)
    
    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()

    # Original event names
    class_names = encoder.inverse_transform(
        np.arange(len(encoder.classes_))
    )

    n_classes = len(class_names)

    # Build 2 x N matrix
    cm = np.zeros((2, n_classes), dtype=int)

    for pred, true in zip(y_pred, y_true):
        cm[pred, true] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(1.2 * n_classes, 4))

    im = ax.imshow(cm, cmap="Blues")

    # Axis labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Pred Non-TDE", "Pred TDE"])

    ax.set_xlabel("Actual Event Type")
    ax.set_ylabel("Prediction")

    # Cache max value to calculate contrast threshold
    cm_max = cm.max()

    # Numbers inside cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            
            # FIX: Base text color on the raw count intensity relative to the max value
            color = "white" if val > (cm_max / 2) else "black"
            
            ax.text(
                j, i,
                str(val),
                ha="center",
                va="center",
                color=color
            )

    plt.colorbar(im, ax=ax)
    plt.title("2 × N Confusion Matrix")
    plt.tight_layout()
    plt.savefig(directory / f"{model_name}_2D_confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.show()
    
def plot_confusion_matrix(y_pred, y_true, label_map, model_name="Model", title="Confusion Matrix", figsize=(10, 8), cmap="Blues"):
    """
    Plots a confusion matrix with human-readable class labels.
    """
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from pathlib import Path

    directory = Path.cwd() / "Results" / "Plots"
    directory.mkdir(parents=True, exist_ok=True)

    # Sorted unique classes present in either split
    classes = sorted(set(np.concatenate([np.unique(y_true), np.unique(y_pred)])))
    labels  = [label_map[c] for c in classes]

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Normalised version for annotation (keep raw counts in heatmap)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,          # we'll write custom annotations below
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    # Custom annotations: count on top, percentage below
    for i in range(len(classes)):
        for j in range(len(classes)):
            count = cm[i, j]
            pct   = cm_norm[i, j] * 100
            color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax.text(
                j + 0.5, i + 0.42,
                f"{count}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=color,
            )
            ax.text(
                j + 0.5, i + 0.62,
                f"({pct:.1f}%)",
                ha="center", va="center",
                fontsize=8, color=color,
            )

    ax.set_xlabel("Predicted label", fontsize=13, labelpad=10)
    ax.set_ylabel("True label",      fontsize=13, labelpad=10)
    ax.set_title(title,              fontsize=15, pad=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(directory / f"{model_name}_confusion_matrix.png", bbox_inches="tight", dpi=300)
    plt.show()
    
def load_data(dir):
    df = pd.read_csv(f"{dir}/train_df.csv")
    df_valid = pd.read_csv(f"{dir}/test_df.csv")
    df_test = pd.read_csv(f"{dir}/CNN_test.csv")
    return df, df_valid, df_test

def drop_unnecessary_columns(data_frames):
    A_lambda = [c for c in data_frames[0].columns if "A_lambda" in c]
    dropped = ["split", "EBV", "Z"] + A_lambda
    for d in data_frames:
        d.drop(columns=dropped, inplace=True)
    return data_frames

def drop_column(dataframes, target):
    for d in dataframes:
        try:
            d.drop(columns=[target], inplace=True)
        except Exception:
            pass
    return dataframes

def evaluate_and_save_metrics_binary(
    csv_filename,
    model,
    history,
    X_train, y_train,
    X_valid, y_valid
):
    """
    Evaluates model on train and validation sets, computes all metrics,
    saves results to CSV, and returns the DataFrame.
    """
    from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, balanced_accuracy_score
    from pathlib import Path

    directory = Path.cwd() / "Results" / "csv results"
    directory.mkdir(parents=True, exist_ok=True)

    # --- Extract best loss and AUCPR from history ---
    min_train_loss = min(history.history["loss"])
    min_val_loss   = min(history.history["val_loss"])
    max_train_auc  = max(history.history["aucpr"])
    max_val_auc    = max(history.history["val_aucpr"])

    # --- Predict ---
    y_train_prob = model.predict(X_train, verbose=0)
    y_valid_prob = model.predict(X_valid, verbose=0)

    # --- Find best threshold + F1 ---
    train_best_f1, train_best_thr = find_best_f1score_and_threshold(y_train_prob, y_train)
    valid_best_f1, valid_best_thr = find_best_f1score_and_threshold(y_valid_prob, y_valid)

    # --- Inner helper ---
    def _compute_metrics(split_name, y_true, y_pred_prob, best_thr, best_f1, min_loss, max_auc):
        y_pred_binary = (y_pred_prob >= best_thr).astype(int)

        recall      = recall_score(y_true, y_pred_binary, zero_division=0)
        specificity = recall_score(y_true, y_pred_binary, pos_label=0, zero_division=0)

        return {
            "Dataset"   : split_name,
            "Min_Loss"  : round(min_loss, 4),
            "Max_AUCPR" : round(max_auc, 4),
            "Threshold" : round(best_thr, 4),
            "F1_Score"  : round(best_f1, 4),
            "MCC"       : round(matthews_corrcoef(y_true, y_pred_binary), 4),
            "Bal_Acc"   : round(balanced_accuracy_score(y_true, y_pred_binary), 4),
            "G_Mean"    : round(np.sqrt(recall * specificity), 4),
            "Precision" : round(precision_score(y_true, y_pred_binary, zero_division=0), 4),
            "Recall"    : round(recall, 4),
        }

    # --- Compute ---
    results_df = pd.DataFrame([
        _compute_metrics("Training",   y_train, y_train_prob, train_best_thr, train_best_f1, min_train_loss, max_train_auc),
        _compute_metrics("Validation", y_valid, y_valid_prob, valid_best_thr, valid_best_f1, min_val_loss,   max_val_auc),
    ])

    # --- Save ---
    results_df.to_csv(directory / csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    return results_df

def set_seed(seed=42):
    import random
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        print("Could not set TensorFlow threading configuration.")
    tf.config.experimental.enable_op_determinism()

    tf.get_logger().setLevel("ERROR")
    tf.keras.backend.clear_session()
    
def plot_metric_curve(history, metric, model_name="Model"):
    from pathlib import Path

    directory = Path.cwd() / "Results" / "Plots"
    directory.mkdir(parents=True, exist_ok=True)

    capital_Names = {'aucpr':'AUCPR', 'macro_auroc':'Macro AUROC', 'loss':'Loss'}
    if metric in ["aucpr","macro_auroc"]:
        func = max
        aim = "max"
    else:
        func = min
        aim = "min"
        
    epochs = range(1, len(history.history[metric]) + 1)
    best_train = func(history.history[metric])
    best_valid = func(history.history[f"val_{metric}"])
    
    print(f"{aim} train {capital_Names[metric]} {round(best_train,3)}")
    print(f"{aim} val {capital_Names[metric]} {round(best_valid,3)}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(epochs, history.history[metric], label=f"Train {capital_Names[metric]}")
    plt.scatter(epochs, history.history[f"val_{metric}"], label=f"Val {capital_Names[metric]}")
    plt.xlabel("Epoch")
    plt.ylabel(f"{capital_Names[metric]}")
    plt.legend()
    plt.savefig(directory / f"{model_name}_{metric}_curve.png", bbox_inches="tight", dpi=300)
    plt.show()
    
def find_best_f1score_and_threshold(
    y_pred_prob,
    y_true,
    binary=True,
    fallback_class=0
):
    from sklearn.metrics import f1_score

    best_f1  = 0
    best_thr = 0
    for thr in np.linspace(0.01, 0.99, 401):
        if binary:
            y_pred = (y_pred_prob > thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        else:
            max_prob = np.max(y_pred_prob, axis=1)
            y_pred = np.argmax(y_pred_prob, axis=1)

            # Reject low-confidence predictions
            y_pred[max_prob < thr] = fallback_class
            f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    return best_f1, best_thr

def evaluate_and_save_metrics_multiclass(
    csv_filename,
    model,
    history,
    X_train, y_train,
    X_valid, y_valid
):
    """
    Evaluates model on train and validation sets, computes all metrics,
    saves results to CSV, and returns the DataFrame.
    """
    from sklearn.metrics import recall_score, precision_score, matthews_corrcoef, balanced_accuracy_score, f1_score
    from sklearn.preprocessing import label_binarize
    from pathlib import Path

    directory = Path.cwd() / "Results" / "csv results"
    directory.mkdir(parents=True, exist_ok=True)

    # --- Extract best loss and AUROC from history ---
    min_train_loss        = min(history.history["loss"])
    min_val_loss          = min(history.history["val_loss"])
    max_train_macro_auroc = max(history.history["macro_auroc"])
    max_val_macro_auroc   = max(history.history["val_macro_auroc"])

    # --- Predict probabilities then convert to class labels via argmax ---
    y_train_prob = model.predict(X_train, verbose=0)
    y_valid_prob = model.predict(X_valid, verbose=0)

    y_train_pred = np.argmax(y_train_prob, axis=1)
    y_valid_pred = np.argmax(y_valid_prob, axis=1)

    num_classes = y_train_prob.shape[1]

    # --- Inner helper ---
    def _compute_metrics(split_name, y_true, y_pred, min_loss, max_auc):
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        y_pred_bin = label_binarize(y_pred, classes=np.arange(num_classes))

        # Per-class recall (sensitivity) and specificity, then macro-average
        per_class_recall      = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_specificity = np.array([
            recall_score(y_true_bin[:, c], y_pred_bin[:, c], pos_label=0, zero_division=0)
            for c in range(num_classes)
        ])
        macro_recall      = per_class_recall.mean()
        macro_specificity = per_class_specificity.mean()

        return {
            "Dataset"         : split_name,
            "Min_Loss"        : round(min_loss, 4),
            "Max_Macro_AUROC" : round(max_auc, 4),
            "Macro_F1"        : round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "Multiclass_MCC"  : round(matthews_corrcoef(y_true, y_pred), 4),
            "Balanced_Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "Macro_G_Mean"    : round(np.sqrt(macro_recall * macro_specificity), 4),
            "Macro_Precision" : round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "Macro_Recall"    : round(macro_recall, 4),
        }

    # --- Compute ---
    results_df = pd.DataFrame([
        _compute_metrics("Training",   y_train, y_train_pred, min_train_loss, max_train_macro_auroc),
        _compute_metrics("Validation", y_valid, y_valid_pred, min_val_loss,   max_val_macro_auroc),
    ])

    # --- Save ---
    results_df.to_csv(directory / csv_filename, index=False)
    print(f"Metrics saved to {csv_filename}")

    return results_df

def labels_weight(y_combined):
    from sklearn.utils import class_weight
    classes = np.unique(y_combined)
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_combined)
    class_weights = dict(zip(classes, weights))
    return class_weights

import tensorflow as tf # Required globally if you want to use the decorator directly on the class block
@tf.keras.utils.register_keras_serializable()
class MacroAUROC(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="macro_auroc", **kwargs):
        super().__init__(name=name, **kwargs)
        import tensorflow as tf
        self.num_classes = num_classes
        self.auc_metrics = [
            tf.keras.metrics.AUC(curve="ROC", name=f"auc_class_{i}")
            for i in range(num_classes)
        ]

    def update_state(self, y_true, y_pred, sample_weight=None):
        import tensorflow as tf
        # One-vs-Rest: evaluate each class independently
        for i, auc in enumerate(self.auc_metrics):
            # Binary mask: is this sample class i or not
            y_true_binary = tf.cast(tf.equal(tf.cast(y_true, tf.int32), i), tf.float32)
            y_pred_i = y_pred[:, i] if self.num_classes > 1 else y_pred
            auc.update_state(y_true_binary, y_pred_i, sample_weight)

    def result(self):
        import tensorflow as tf
        return tf.reduce_mean([auc.result() for auc in self.auc_metrics])

    def reset_state(self):
        for auc in self.auc_metrics:
            auc.reset_state()