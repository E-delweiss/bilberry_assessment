import logging

import torch
import numpy as np

import metrics

def validation_loop(model, validation_dataset, criterion, DO_VALIDATION:bool)->float:
    """
    Execute validation loop. 
    Since the validation dataset is small, we use data augmentation.
    Data augmentation involve looping over validation_dataset multiple times.

    Args:
        model (torch.nn.Module): current model to validate
        validation_dataset (torch.utils.data.Dataset): validation dataset
        criterion (torch.nn.modules.loss): loss function
        DO_VALIDATION (bool): do validation loop. If not, accuracy is set to 999
        VAL_EPOCHS (int): loop over validation dataset

    Returns:
        val_loss (float): validation loss as the mean of all batch losses
        val_acc (float): validation accuracy as the mean of all batch accuracies.
    """
    if DO_VALIDATION:
        logging.info(f"Start validation loop")

        ### Set model to validation state
        model.eval()
        
        ### Set device as the model device
        device = next(model.parameters()).device

        batch_acc = []
        batch_loss = []
        for (img, target) in validation_dataset:
            img, target = img.to(device), target.to(device)

            ### Disable autograd for validation
            with torch.no_grad():
                ### prediction
                prediction = model(img).squeeze(1)

                ### loss 
                loss = criterion(prediction, target)
                batch_loss.append(loss.item())

            ### Compute and save accuracy for each batch
            acc = metrics.class_acc(target, prediction)
            batch_acc.append(acc)
            

        ### Compute validation accuracy and loss
        val_acc = np.mean(batch_acc)
        val_loss = np.mean(batch_loss)
        confmatrix_dict = metrics.metrics(target, prediction)

    else:
        logging.warning(f"Validation loop disabled. Validation accuracy and loss are set to 999")
        val_acc = 999
        val_loss = 999
        confmatrix_dict = {}

    return val_loss, val_acc, confmatrix_dict



