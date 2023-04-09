import os, sys
import datetime
import logging
from configparser import ConfigParser

from torch.utils.tensorboard import SummaryWriter
import torch
from icecream import install, ic
install()

import utils
import modelBuilder
from bilberry_dataset import get_training_dataset, get_validation_dataset
from metrics import class_acc, metrics
from validation import validation_loop

### Set time format
################################################################################
delta_time = datetime.timedelta(hours=1)
timezone = datetime.timezone(offset=delta_time)
time_formatted = datetime.datetime.now(tz=timezone)
time_formatted = '{:%Y-%m-%d %H:%M:%S}'.format(time_formatted)

### Handle subfolder paths
################################################################################
current_folder = os.path.dirname(locals().get("__file__"))
config_file = os.path.join(current_folder, "config.ini")
sys.path.append(config_file)

### Load config file
################################################################################
config = ConfigParser()
config.read('config.ini')

### Read config file variables
DEVICE = config.get('TRAINING', 'device')
learning_rate = config.getfloat('TRAINING', 'learning_rate')
BATCH_SIZE = config.getint('TRAINING', 'batch_size')
WEIGHT_DECAY = config.getfloat('TRAINING', 'weight_decay')
DO_VALIDATION = config.getboolean('TRAINING', 'do_validation')
EPOCHS = config.getint('TRAINING', 'nb_epochs')
LR_SCHEDULER = config.getboolean('TRAINING', 'lr_scheduler')

PREFIX = config.get('MODEL', 'model_name')
PRETRAINED = config.getboolean('MODEL', 'pretrained')

TRAIN_RATIO = config.getint('DATASET', 'train_ratio')
VAL_RATIO = config.getint('DATASET', 'val_ratio')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

FREQ = config.getint('PRINTING', 'freq')
SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

### Training setup
################################################################################
device = utils.set_device(DEVICE, verbose=0)

# model = modelBuilder.resNetBilberry(pretrained=PRETRAINED)
model = modelBuilder.efficientNetBilberry(pretrained=PRETRAINED)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.BCELoss()

train_dataloader = get_training_dataset(BATCH_SIZE, ratio=TRAIN_RATIO, isAugment=isAugment_trainset)
val_dataloader = get_validation_dataset(ratio=VAL_RATIO, isAugment=isAugment_valset)

### Initialize log file
################################################################################
print(f"\n\n[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

writer = SummaryWriter(log_dir=f'runs/{PREFIX}_RUN_{time_formatted}')
utils.create_logging(prefix=PREFIX)
logging.info(f"Pretrained is {PRETRAINED}")
logging.info(f"Learning rate = {learning_rate}")
logging.info(f"Batch size = {BATCH_SIZE}")
logging.info(f"Using optimizer : {optimizer}")
logging.info("Lr Scheduler : None")
logging.info("")
logging.info("Start training")
logging.info(f"[START] : {time_formatted}")


### Training / Validation loops
################################################################################
train_loss_epoch = []
train_loss_batch = []
train_acc_epoch = []
train_acc_batch = []

val_acc_epoch = []
val_acc_batch = []

it = 0 
start_time = datetime.datetime.now()
for epoch in range(EPOCHS):
    epochs_loss = 0.
    
    print(" "*5 + f"EPOCH {epoch+1}/{EPOCHS}")
    print(" "*5 + f"Learning rate : lr = {optimizer.defaults['lr']}")
    for batch, (img, target) in utils.tqdm_fct(train_dataloader):
        model.train()
        loss = 0
        img, target = img.to(device), target.to(device)
        
        ### clear gradients
        optimizer.zero_grad()
        
        ### prediction
        prediction = model(img).squeeze(1)

        ### compute loss in the current batch
        loss = criterion(prediction, target)

        ### compute gradients
        loss.backward()
        
        ### Weight updates
        optimizer.step()

        ### Class accuracy
        train_acc = class_acc(target, prediction)

        ### Record loss
        current_loss = loss.item()
        epochs_loss += current_loss

        if batch == 0 or (batch+1)%FREQ == 0 or batch==len(train_dataloader.dataset)//BATCH_SIZE:
            # Recording training metrics
            writer.add_scalars('variables', {f"Loss/train" : current_loss}, it)
            writer.add_scalars('variables', {f"Acc/train" : train_acc}, it)

            # Pretty print
            utils.pretty_print(batch, BATCH_SIZE, len(train_dataloader.dataset), current_loss, train_acc)

            ### Validation loop
            val_loss, val_acc, confmatrix_dict = validation_loop(model, val_dataloader, criterion, DO_VALIDATION)
            writer.add_scalars('variables', {f"Loss/val" : val_loss}, it)
            writer.add_scalars('variables', {f"Acc/val" : val_acc}, it)
            writer.add_scalars('variables', {f"F1/val" : confmatrix_dict["F1_score"]}, it)

            print(f"** Validation loss : {val_loss:.5f}")
            print(f"** Validation accuracy : {val_acc*100:.2f}%")
            it += 1
            # Write logs
            if batch == len(train_dataloader.dataset)//BATCH_SIZE:
                print(f"Mean training loss for this epoch : {epochs_loss / len(train_dataloader):.5f} \n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"** Training loss : {epochs_loss / len(train_dataloader):.5f}")
                logging.info(f"** Validation loss : {val_loss:.5f}")
                logging.info(f"***** Training acc : {train_acc*100:.2f}%")
                logging.info(f"***** Validation acc : {val_acc*100:.2f}%")




### Saving results
################################################################################
utils.save_model(model, PREFIX, epoch, SAVE_MODEL)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
logging.shutdown
