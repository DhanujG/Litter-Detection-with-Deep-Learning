import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchaudio
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


def random_crop(input):
    # TODO: randomly crop the image
    return input, has_trash


class TrashModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.feat_extractor = None # TODO: use pretrained imagenet
        self.l1 = torch.nn.Linear(100, 1)
        self.lossfn = F.mse_loss

    def forward(self, x):
        return torch.relu(self.l1(x))

    def training_step(self, batch, batch_idx):
        image, _, _ = batch
        cropped_image, has_trash = random_crop(image)
        pred = self(cropped_image)
        loss = self.lossfn(pred, has_trash)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, _, _ = batch
        cropped_image, has_trash = random_crop(image)
        pred = self(cropped_image)
        loss = self.lossfn(pred, has_trash)
        return {'val_loss': self.lossfn(pred, has_trash)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



trash_data = None # TODO: add dataset

train_dataloader = DataLoader(trash_data, batch_size=32, num_workers=4, shuffle=True)
val_dataloader = DataLoader(trash_data, batch_size=32, num_workers=4)


model = TrashModel()

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=True,
   mode='min'
)

trainer = Trainer(show_progress_bar=True, early_stop_callback=early_stop_callback)
trainer.fit(model, train_dataloader, val_dataloader)
