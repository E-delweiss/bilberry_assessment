import torch
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
    labels_pred = torch.argmax(prediction, dim=-1)
    # print("\n")
    # ic(prediction)
    # ic(target)
    
    ### Mean of the right predictions
    acc = (1/BATCH_SIZE) * torch.sum(target == torch.gt(prediction, 0.5)).item()
    return acc