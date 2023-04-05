import logging
from datetime import datetime
import glob

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

def create_logging(prefix:str):
    """
    Create logging template.

    Args:
        prefix (str): base name of logging.
    """
    assert type(prefix) is str, TypeError

    ### Define format
    log_format = (
    '%(asctime)s ::%(levelname)s:: %(message)s'
    )

    ### Define time of creation
    tm = datetime.now()
    tm = tm.strftime("%d%m%Y_%Hh%M")

    ### Set file name
    logging_name = 'logging_'+prefix+'_'+tm+'.log'

    ### Set logging infos
    logging.basicConfig(
        level=logging.INFO,
        format=log_format, datefmt='%d/%m/%Y %H:%M:%S',
        filemode="w",
        filename=(logging_name),
    )
    logging.info("Model is {}.".format(prefix))

def device(verbose:int=0)->torch.device:
    """
    Set the device to 'cpu', 'cuda' or 'mps'.

    Args:
        verbose (int, optional): Display infos. Defaults to 0.

    Returns:
        torch.device: hardware used.
    """

    ### Choosing device between CPU, GPU or MPS
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.has_mps:
        device=torch.device('mps')
    else:
        device=torch.device('cpu')
    
    logging.info("Execute on {}".format(device))
    if verbose:
        print("\n------------------------------------")
        print(f"Execute script on - {device} -")
        print("------------------------------------\n")

    return device


def pretty_print(batch:int, BATCH_SIZE:int, len_training_ds:int, current_loss:float, losses:dict, train_classes_acc:float):
    """
    Print all training infos for the current batch.

    Args:
        batch (int)
            Current batch.
        BATCH_SIZE (int)
            Len of the batch size
        len_training_ds (int)
            Len of the training dataset.
        current_loss (float)
            Current training loss.
        losses (dict)
            Dict of all the losses used to compute the main loss. It contains floats :
            ['loss_xy', 'loss_wh', 'loss_conf_obj', 'loss_conf_noobj', 'loss_class'].
        train_classes_acc (float)
            Training class accuracy.
    """
    BATCH_SIZE = 128
    if batch+1 <= len_training_ds//BATCH_SIZE:
        current_training_sample = (batch+1)*BATCH_SIZE
    else:
        current_training_sample = batch*BATCH_SIZE + len_training_ds%BATCH_SIZE

    print(f"\n--- Image : {current_training_sample}/{len_training_ds}")
    print(f"* loss = {current_loss:.5f}")
    print(f"* xy_coord training loss for this batch : {losses['loss_xy']:.5f}")
    print(f"* wh_sizes training loss for this batch : {losses['loss_wh']:.5f}")
    print(f"* confidence with object training loss for this batch : {losses['loss_conf_obj']:.5f}")
    print(f"* confidence without object training loss for this batch : {losses['loss_conf_noobj']:.5f}")
    print(f"* class training loss for this batch : {losses['loss_class']:.5f}")
    print("\n")
    print(f"** Training class accuracy : {train_classes_acc*100:.2f}%")


def update_lr(current_epoch:int, optimizer:torch.optim):
    """
    Schedule the learning rate

    Args:
        current_epoch (int): Current training loop epoch.
        optimizer (torch.optim): Gradient descent optimizer.
    """
    if current_epoch > 7:
        optimizer.defaults['lr'] = 0.0001


def save_model(model, path:str, save:bool):
    """
    Handle torch model saving.

    Args:
        model (): torch model
        path (str): saving path .pt
        save (bool): save the model or not
    """
    if save:
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        path = path+'_'+tm+'.pt'
        torch.save(model.state_dict(), path)
        logging.info("\nModel saved to {}.".format(path))
        print("*"*5, "\nModel saved to {}.".format(path))
    return

def save_losses(train_loss:dict, val_loss:dict, model_name:str, save:bool):
    """
    Save training en validation losses to pickle files.

    Args:
        train_loss (dict)
        val_loss (dict)
        model_name (str)
        save (bool)
    """
    if save:
        tm = datetime.now()
        tm = tm.strftime("%d%m%Y_%Hh%M")
        train_path = f"train_results_{model_name}_{tm}.pkl"
        val_path = f"val_results_{model_name}_{tm}.pkl"
        
        with open(train_path, 'wb') as pkl:
            pickle.dump(train_loss, pkl)

        with open(val_path, 'wb') as pkl:
            pickle.dump(val_loss, pkl)
        
        logging.info("Training results saved to {}.".format(train_path))
        logging.info("Validation results saved to {}.".format(val_path))
    else:
        logging.warning("No saving has been requested for losses.")
    return

def tqdm_fct(training_dataset):
    """
    Display a nice progress bar when training.

    Args:
        training_dataset (torch.dataset): training dataloader

    Returns:
        None.
    """
    return tqdm(enumerate(training_dataset),
                total=len(training_dataset),
                initial=1,
                desc="Training : image",
                ncols=100)


def mean_std_normalization()->tuple:
    """
    DONE
    Get the mean and std of the dataset RGB channels.
    mean/std of the whole dataset :
        mean=(0.4551, 0.4672, 0.4151), std=(0.2522, 0.2451, 0.2808)

    Returns:
        mean : torch.Tensor
        std : torch.Tensor
    """
    data_jpg = glob.glob('dataset/fields/*')
    data_jpg = data_jpg + glob.glob('dataset/roads/*')
    data_PIL = [Image.open(img_path).convert('RGB') for img_path in data_jpg]
    data_tensor = [torchvision.transforms.ToTensor()(img_PIL) for img_PIL in data_PIL]

    channels_sum, channels_squared_sum = 0, 0
    for img in data_tensor:
        channels_sum += torch.mean(img, dim=[1,2])
        channels_squared_sum += torch.mean(img**2, dim=[1,2])
    
    mean = channels_sum/len(data_tensor)
    std = torch.sqrt((channels_squared_sum/len(data_tensor) - mean**2))
    return mean, std



if __name__ == "__main__":
    mean, std = mean_std_normalization()
    print(mean)
    print(std)


