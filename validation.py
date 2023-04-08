import logging

import torch
import numpy as np

import metrics

def validation_loop(model, validation_dataset, DO_VALIDATION:bool, VAL_EPOCHS:int=1)->float:
    """
    Execute validation loop. 
    Since the validation dataset is small, we use data augmentation.
    Data augmentation involve looping over validation_dataset multiple times.

    Args:
        model (torch.nn.Module): current model to validate
        validation_dataset (torch.utils.data.Dataset): validation dataset
        DO_VALIDATION (bool): do validation loop. If not, accuracy is set to 999
        VAL_EPOCHS (int): loop over validation dataset

    Returns:
        val_acc (float): validation accuracy as the mean of all batch accuracies.
    """
    if DO_VALIDATION:
        logging.info(f"Start validation loop")

        ### Set model to validation state
        model.eval()
        
        ### Set device as the model device
        device = next(model.parameters()).device

        batch_acc = []
        for epoch in range(VAL_EPOCHS):
            for (img, target) in validation_dataset:
                img, target = img.to(device), target.to(device)

                ### Disable autograd for validation
                with torch.no_grad():
                    ### prediction
                    prediction = model(img).squeeze(1)
                
                ### Compute and save accuracy for each batch and each epoch
                acc = metrics.class_acc(target, prediction)
                batch_acc.append(acc)

        ### Compute validation accuracy
        val_acc = np.mean(batch_acc)

    else:
        logging.warning(f"Validation loop disabled. Validation accuracy set to 999")
        val_acc = 999

    return val_acc



