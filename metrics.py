import torch
import numpy as np
from icecream import ic

def class_acc(target:torch.Tensor, prediction:torch.Tensor)->float:
    """
    Compute model accuracy

    Args:
        target (torch.Tensor) size : TODO
        prediction (torch.Tensor) size : TODO

    Returns:
        float: accuracy in (%)
    """
    BATCH_SIZE = len(target)
    
    ### Mean of the right predictions
    acc = (1/BATCH_SIZE) * torch.sum(target == torch.greater_equal(prediction, 0.5)).item()
    return acc


def metrics(target:torch.Tensor, prediction:torch.Tensor)->dict:
    """
    TODO
    """
    # prediction = prediction.to(torch.device("cpu"))

    ### Compute confusion matrix
    TP = (torch.round(prediction) == 1) & (target == 1)
    TN = (torch.round(prediction) == 0) & (target == 0)
    FP = (torch.round(prediction) == 1) & (target == 0)
    FN = (torch.round(prediction) == 0) & (target == 1)
    
    count_TP = TP.sum()
    count_TN = TN.sum()
    count_FP = FP.sum()
    count_FN = FN.sum()    
    
    ### Compute precision, recall and F1score
    precision = count_TP / (count_TP + count_FP + 1e-6)
    recall = count_TP / (count_TP + count_FN + 1e-6)
    F1_score = 2*(precision*recall) / (precision + recall + 1e-6)

    metrics_dict = {
        "TP" : count_TP,
        "TN" : count_TN, 
        "FP" : count_FP,
        "FN" : count_FN,
        "recall" : recall,
        "precision" : precision,
        "F1_score" : F1_score
    }
    return metrics_dict