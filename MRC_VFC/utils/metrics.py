import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score


def compute_avg_metrics(groundTruth, activations):
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    mean_acc = accuracy_score(y_true=groundTruth, y_pred=predictions)
    f1_macro = f1_score(y_true=groundTruth, y_pred=predictions, average='macro')
    try:
        auc = roc_auc_score(y_true=groundTruth, y_score=activations, multi_class='ovr')
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    bac = balanced_accuracy_score(y_true=groundTruth, y_pred=predictions)
    sens_macro = sensitivity_score(y_true=groundTruth, y_pred=predictions, average='macro')
    spec_macro = specificity_score(y_true=groundTruth, y_pred=predictions, average='macro')

    return mean_acc, f1_macro, auc, bac, sens_macro, spec_macro


def compute_per_class_metrics(groundTruth, activations, num_classes=None):
    """
    Compute one-vs-rest metrics for each class.
    Returns a list of dicts:
      class_id, support, acc(=recall), f1, auc, bac, sens, spec, precision
    """
    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)

    if num_classes is None:
        num_classes = int(activations.shape[1])
    labels = list(range(int(num_classes)))
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions, labels=labels)
    total = float(cm.sum())

    per_class = []
    for c in labels:
        tp = float(cm[c, c])
        fn = float(cm[c, :].sum() - tp)
        fp = float(cm[:, c].sum() - tp)
        tn = float(total - tp - fn - fp)

        support = int(tp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * precision * sens / (precision + sens)) if (precision + sens) > 0 else 0.0
        bac = 0.5 * (sens + spec)

        y_true_bin = (groundTruth == c).astype(np.int32)
        y_score = activations[:, c]
        if y_true_bin.min() == y_true_bin.max():
            auc = float("nan")
        else:
            try:
                auc = float(roc_auc_score(y_true=y_true_bin, y_score=y_score))
            except ValueError:
                auc = float("nan")

        per_class.append(
            {
                "class_id": int(c),
                "support": support,
                "acc": float(sens),
                "f1": float(f1),
                "auc": float(auc),
                "bac": float(bac),
                "sens": float(sens),
                "spec": float(spec),
                "precision": float(precision),
            }
        )
    return per_class


def compute_confusion_matrix(groundTruth, activations, labels):

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions, labels=labels)

    return cm


def epochVal(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

        acc, f1, auc, bac, sens, spec = compute_avg_metrics(groundTruth, activations)

    model.train(training)

    return acc, f1, auc, bac, sens, spec


def epochTest(model, dataLoader):
    training = model.training
    model.eval()

    groundTruth = torch.Tensor().cuda()
    activations = torch.Tensor().cuda()

    with torch.no_grad():
        for i, (image, label) in enumerate(dataLoader):
            image, label = image.cuda(), label.cuda()
            output = model(image)
            if isinstance(output, tuple):
                _, output = output
            output = F.softmax(output, dim=1)
            groundTruth = torch.cat((groundTruth, label))
            activations = torch.cat((activations, output))

    groundTruth = groundTruth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)
    cm = confusion_matrix(y_true=groundTruth, y_pred=predictions)
    model.train(training)

    return cm
