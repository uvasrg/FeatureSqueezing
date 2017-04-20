import sklearn
import numpy as np
from sklearn.metrics import roc_curve, auc
import pdb

def get_tpr_fpr(true_labels, pred, threshold):
    pred_labels = pred > threshold
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    return TP/np.sum(true_labels), FP/np.sum(1-true_labels)


def get_roc_data_val(l1_dist, nb_samples = 1000, nb_cols=2):
    x_train = np.hstack( [  l1_dist[:nb_samples/2,i] for i in range(nb_cols)])
    y_train = np.hstack([np.zeros(nb_samples/2), np.ones(nb_samples/2*(nb_cols-1))])

    x_val = np.hstack( [  l1_dist[nb_samples/2:,i] for i in range(nb_cols)])
    y_val = y_train


def train_detector(x_train, y_train, x_val, y_val):
    fpr, tpr, thresholds = roc_curve(y_train, x_train)
    accuracy = [ sklearn.metrics.accuracy_score(y_train, x_train>threshold, normalize=True, sample_weight=None) for threshold in thresholds ]
    roc_auc = auc(fpr, tpr)

    idx_best = np.argmax(accuracy)
    print "Best training accuracy: %.4f, TPR(Recall): %.4f, FPR: %.4f @%.4f" % (accuracy[idx_best], tpr[idx_best], fpr[idx_best], thresholds[idx_best])
    print "ROC_AUC: %.4f" % roc_auc

    accuracy_val = [ sklearn.metrics.accuracy_score(y_val, x_val>threshold, normalize=True, sample_weight=None) for threshold in thresholds ]
    tpr_val, fpr_val = zip(*[ get_tpr_fpr(y_val, x_val, threshold)  for threshold in thresholds  ])
    # roc_auc_val = auc(fpr_val, tpr_val)
    print "Validation accuracy: %.4f, TPR(Recall): %.4f, FPR: %.4f @%.4f" % (accuracy_val[idx_best], tpr_val[idx_best], fpr_val[idx_best], thresholds[idx_best])

    return threshold, accuracy_val, fpr_val, tpr_val