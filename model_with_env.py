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




class TrashModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.feat_extractor = models.alexnet(pretrained=True)
        self.l1 = torch.nn.Linear(1000, 8)
        self.lossfn = F.mse_loss
        self.test_size = 0
        self.test_sum = 0

    def forward(self, x):
        return torch.sigmoid(self.l1(torch.relu(self.feat_extractor(x))))

    def training_step(self, batch, batch_idx):
        image, has_trash, env = batch
        trash = has_trash.float().unsqueeze(1)

        pred = self(image)
        pred_class, pred_env = torch.split(pred, [1, 7], dim=1)
        pred_env = torch.argmax(pred_env, dim=1)

        loss = self.lossfn(pred_class, trash) + self.lossfn((pred_env == env), 1)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        image, has_trash, env = batch
        trash = has_trash.float().unsqueeze(1)
        pred = self(image)
        pred_class, pred_env = torch.split(pred, [1, 7], dim=1)
        pred_env = torch.argmax(pred_env, dim=1)

        loss = self.lossfn(pred_class, trash) + self.lossfn((pred_env == env), 1)

        # split pred into classification part
        # and env prediction part

        out_class = (pred_class > 0.5).float()

        accuracy_class = (torch.sum(out_class == trash)) / (len(trash)*1.0)
        accuracy_env = torch.sum(pred_env == env, dim=0) / (len(trash)*1.0)

        # self.test_size += 1
        # self.test_sum = accuracy + self.test_sum

        tensorboard_logs = {'test_loss': loss, 'test_accuracy_class': accuracy_class, 'test_accuracy_env' : accuracy_env}

        return {'loss': loss, 'accuracy_class': accuracy_class, 'accuracy_env': accuracy_env, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_idx):
        image, has_trash, env = batch
        trash = has_trash.float().unsqueeze(1)

        # pred's 8 outputs are 1 classifier, and then 7 env vals
        pred = self(image)
        pred_class, pred_env = torch.split(pred, [1, 7], dim=1)

        #print(pred_env)
        pred_env = torch.argmax(pred_env, dim=1)

        #print(pred_env == env)

        loss1 = self.lossfn(pred_class, trash)


        loss2 = self.lossfn((pred_env == env), 1)

        loss = loss1 + loss2

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



trash_data = CroppedTrashDataset("./new_data.txt", "./new_data/")





train_dataloader = DataLoader(trash_data, batch_size=32, shuffle=True, collate_fn = mod_collate)
val_dataloader = DataLoader(trash_data, batch_size=100, shuffle = True, collate_fn = mod_collate)





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
trainer.fit(model, train_dataloader, val_dataloader)



#TEST MODELS
#test the model
#trainer.test(model = model, test_dataloaders = val_dataloader, ckpt_path = "./lightning_logs/version_7/checkpoints/epoch=6-step=216.ckpt")
# print(model.test_sum/ model.test_size)


