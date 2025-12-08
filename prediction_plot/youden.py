from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import numpy as np


def youden_index(
        y_true,
        y_score,
        pos_label=1,
        thr_start=0.0,      # 阈值起点
        thr_end=0.505,      # 阈值终点
        thr_step=0.005      # 阈值步长
    ):
    """
    clinical indicators: PPV, NPV, Sensitivity, Specificity, Youden Index, F1, TP FP TN FN
    YoudenIndex = sensitivity + specificity - 1

    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_score : array-like
        Predicted scores (probability)
    pos_label : int
        Positive class label
    thr_start : float
        Start of threshold range
    thr_end : float
        End of threshold range (inclusive)
    thr_step : float
        Step size of threshold

    Returns
    -------
    df : DataFrame
        Metrics for each threshold
    max_ji_val : float
        Maximum Youden index
    max_f1_val : float
        Maximum F1 score
    roc_auc : float
        AUC value
    """

    # Compute ROC AUC
    fpr, tpr, thr_raw = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    # Build threshold list
    thrs = np.arange(thr_start, thr_end + 1e-9, thr_step)

    # Prepare result list
    rows = []

    for thr in thrs:
        # Predict using threshold
        y_pred = (y_score >= thr).astype(int)

        # Confusion matrix: tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Sensitivity, Specificity
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        # PPV NPV
        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0

        # Accuracy
        acc = (tp + tn) / (tp + tn + fp + fn)

        # Youden index
        youden = sens + spec - 1

        # F1 score
        f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) else 0.0

        rows.append([
            thr, acc, ppv, npv, sens, spec, youden, f1,
            tp + fn,        # True positive count
            fp + tn,        # True negative count
            tp + fp,        # Predicted positive
            tn + fn,        # Predicted negative
            tp, fp, tn, fn
        ])

    columns = [
        "Thr", "ACC", "PPV", "NPV", "Sens(Rec/TPR)", "Spec",
        "YoudenIdx", "F1",
        "TruePositives", "TrueNegatives",
        "PredPos", "PredNeg",
        "TP", "FP", "TN", "FN"
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Compute maxima
    max_ji_val = df["YoudenIdx"].max()
    max_f1_val = df["F1"].max()

    return df, max_ji_val, max_f1_val, roc_auc
