import torch
import metrics

def validation_loop(model, validation_dataset, DO_VALIDATION, VAL_EPOCHS=10):
    """
    _summary_

    Args:
        model (_type_): _description_
        validation_dataset (_type_): _description_
        ONE_BATCH (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    model.eval()
    device = next(model.parameters()).device
    for epoch in VAL_EPOCHS:
        
        for (img, target) in validation_dataset:
            img, target = img.to(device), target.to(device)
            
            with torch.no_grad():
                ### prediction
                prediction = model(img)
                
                if ONE_BATCH is True:
                    break

    return img, target, prediction



