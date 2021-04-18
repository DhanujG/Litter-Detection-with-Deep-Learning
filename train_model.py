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


def get_info(filename):
    cut_filename = filename[:-4]
    l = cut_filename.split("_")
    id, crop, trash, env = l
    env_list = [int(x) for x in env]
    return int(id), int(crop), trash == "1", env_list

class TrashModel(LightningModule):

    def __init__(self):
        super().__init__()
        self.feat_extractor = None # TODO: use pretrained imagenet
        self.l1 = torch.nn.Linear(100, 1)
        self.lossfn = F.mse_loss

    def forward(self, x):
        return torch.relu(self.l1(x))

    def training_step(self, batch, batch_idx):
        image, has_trash, env_list = batch
        trash = if has_trash 1 else 0
        pred = self(image)
        loss = self.lossfn(pred, trash)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image, has_trash, env_list = batch
        trash = if has_trash 1 else 0
        pred = self(image)
        loss = self.lossfn(pred, trash)
        return {'val_loss': self.lossfn(pred, has_trash)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)



class CroppedTrashDataset(Dataset):
    def __init__(self, metafile, folder):
        self.folder = folder
        self.file_list = []
        with open(metafile) as mf:
            for filename in mf.read().splitlines():
                self.file_list.append(filename)

    def __len__(self):
        return len(file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        id, crop, trash, env_list = get_info(filename)
        img = Image.open(os.path.join(self.folder, filename)) # PIL Image
        img_tensor = ToTensor()(img)
        return img_tensor, trash, env_list





trash_data = CroppedTrashDataset("./new_data.txt", "./new_data")

train_dataloader = DataLoader(trash_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(trash_data, batch_size=32)


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
