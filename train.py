import os, sys
import datetime
import logging
from configparser import ConfigParser

import torch
from icecream import install, ic
install()

import utils
import modelBuilder
from bilberry_dataset import get_training_dataset, get_validation_dataset
from metrics import class_acc
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

isNormalize_trainset = config.getboolean('DATASET', 'isNormalize_trainset')
isAugment_trainset = config.getboolean('DATASET', 'isAugment_trainset')
isNormalize_valset = config.getboolean('DATASET', 'isNormalize_valset')
isAugment_valset = config.getboolean('DATASET', 'isAugment_valset')

FREQ = config.getint('PRINTING', 'freq')
SAVE_MODEL = config.getboolean('SAVING', 'save_model')
SAVE_LOSS = config.getboolean('SAVING', 'save_loss')

### Training setup
################################################################################
device = utils.set_device(DEVICE, verbose=0)

model = modelBuilder.resNetBilberry(pretrained=PRETRAINED)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.BCELoss()

train_dataloader = get_training_dataset(BATCH_SIZE, ratio=1, isNormalize=isNormalize_trainset, isAugment=isAugment_trainset)
val_dataloader = get_validation_dataset(ratio=1, isNormalize=isNormalize_valset, isAugment=isAugment_valset)

### Initialize log file
################################################################################
print(f"[Training on] : {str(device).upper()}")
print(f"Learning rate : {optimizer.defaults['lr']}")

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
train_loss_list = []
train_acc_list = []

val_acc_list = []

start_time = datetime.datetime.now()
for epoch in range(EPOCHS):
    # utils.update_lr(epoch, optimizer, LR_SCHEDULER)
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
            # Recording the total loss
            train_loss_list.append(current_loss)

            # Recording training accuracy
            train_acc_list.append(train_acc)

            # Pretty print
            utils.pretty_print(batch, BATCH_SIZE, len(train_dataloader.dataset), current_loss, train_acc)

            ### Validation loop
            val_acc = validation_loop(model, val_dataloader, DO_VALIDATION)
            val_acc_list.append(val_acc)

            print(f"** Validation accuracy : {val_acc*100:.2f}%")

            # Write logs
            if batch == len(train_dataloader.dataset)//BATCH_SIZE:
                print(f"Mean training loss for this epoch : {epochs_loss / len(train_dataloader):.5f} \n\n")
                logging.info(f"Epoch {epoch+1}/{EPOCHS}")
                logging.info(f"***** Training loss : {epochs_loss / len(train_dataloader):.5f}")
                logging.info(f"***** Training acc : {train_acc*100:.2f}%")
                logging.info(f"***** Validation acc : {val_acc*100:.2f}%")




### Saving results
################################################################################
pickle_val_results = {
"batch_val_class_acc":val_acc_list
}

pickle_train_results = {
    "batch_train_class_acc":train_acc_list,
}

utils.save_model(model, PREFIX, epoch, SAVE_MODEL)
utils.save_losses(pickle_train_results, pickle_val_results, PREFIX, SAVE_LOSS)

end_time = datetime.datetime.now()
logging.info('Time duration: {}.'.format(end_time - start_time))
logging.info("End training.")
logging.shutdown
