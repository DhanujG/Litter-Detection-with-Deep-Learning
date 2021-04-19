import os

import torch
from torchvision.transforms import ToTensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from PIL import Image
import numpy as np
import re
import collections
from torch._six import string_classes
from sklearn import metrics
import torchvision.models as models

from dataset import *




def get_info(filename):
    cut_filename = filename[:-4]
    l = cut_filename.split("_")
    id, crop, trash, env = l
    env_list = [int(x) for x in env]
    return int(id), int(crop), trash == "1", env_list

class TrashModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.feat_extractor = models.alexnet(pretrained=True)
        self.l1 = torch.nn.Linear(1000, 1)
        self.lossfn = F.mse_loss
        self.test_size = 0
        self.test_sum = 0

    def forward(self, x):
        return torch.relu(self.l1(torch.relu(self.feat_extractor(x))))

    def training_step(self, batch, batch_idx):
        image, has_trash, env_list = batch
        trash = has_trash.float().unsqueeze(1)
        pred = self(image)
        # print(pred.shape, trash.shape)
        loss = self.lossfn(pred, trash)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        image, has_trash, env_list = batch
        trash = has_trash.float().unsqueeze(1)
        pred = self(image)
        loss = self.lossfn(pred, trash)
        #accuracy = metrics.accuracy_score(trash, pred)
        #t = Variable(torch.Tensor([0.5]))
        out = (pred > 0.5).float()
        accuracy = (torch.sum(out == trash))/ (len(trash)*1.0)
        self.test_size =+1
        self.test_sum = accuracy + self.test_sum
        tensorboard_logs = {'test_loss': loss, 'test_accuracy': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        image, has_trash, env_list = batch
        trash = has_trash.float().unsqueeze(1)
        pred = self(image)
        loss = self.lossfn(pred, trash)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



trash_data = CroppedTrashDataset("./new_data.txt", "./new_data/")





train_dataloader = DataLoader(trash_data, batch_size=32, shuffle=True, collate_fn = mod_collate)
val_dataloader = DataLoader(trash_data, batch_size=32, collate_fn = mod_collate)





model = TrashModel()





early_stop_callback = EarlyStopping(
   monitor='train_loss',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='min'
)



#TRAIN MODEL
#trainer = Trainer(show_progress_bar=True, early_stop_callback=early_stop_callback)
trainer = Trainer(logger=True, gpus=1, callbacks=[early_stop_callback])
#trainer.fit(model, train_dataloader, val_dataloader)



#TEST MODEL
#test the model
trainer.test(model = model, test_dataloaders = val_dataloader, ckpt_path = "./lightning_logs/version_6/checkpoints/epoch=11-step=491.ckpt")
print(model.test_sum/ model.test_size)


