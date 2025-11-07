from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import numpy as np

def get_confusion_matrix_values(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm[0][0], cm[0][1], cm[1][0], cm[1][1]

class Metrics:
    def __init__(self, y_true, y_pred, y_pred_prob=None):
        """
        Metrics class for evaluating classification performance.

        Parameters:
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.
        - y_pred_prob: Predicted probabilities for the positive class (needed for ROC-AUC and mAP).
        """
        self.y_true = y_true 
        self.y_pred = y_pred 
        self.y_pred_prob = y_pred_prob
        self.tn, self.fp, self.fn, self.tp = get_confusion_matrix_values(y_true, y_pred)

    def g_mean(self):
        sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0
        return np.sqrt(sensitivity * specificity)

    def f1_score(self):
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    def mcc(self):
        nu = (self.tp * self.tn) - (self.fp * self.fn)
        de = np.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return nu / de if de > 0 else 0

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0

    def roc_auc(self):
        unique_classes = np.unique(self.y_true)
        if len(unique_classes) < 2:
            return None
        return roc_auc_score(self.y_true, self.y_pred_prob)

    def mean_average_precision(self):
        if self.y_pred_prob is None or len(np.unique(self.y_true)) < 2:
            return None
        return average_precision_score(self.y_true, self.y_pred_prob)

    def all_metrics(self):
        metrics = {
            "G-mean": self.g_mean(),
            "F1-score": self.f1_score(),
            "MCC": self.mcc()
        }
        if self.y_pred_prob is not None:
            auc = self.roc_auc()
            ap = self.mean_average_precision()
            if auc is not None:
                metrics["AUROC"] = auc
            if ap is not None:
                metrics["mAP"] = ap
        return metrics
