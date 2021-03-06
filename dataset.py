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







class CroppedTrashDataset(Dataset):
    def get_info(self, filename):
        cut_filename = filename[:-4]
        l = cut_filename.split("_")
        id, crop, trash, env = l
        return int(id), int(crop), trash == "1", int(env)-1

    def __init__(self, metafile, folder):
        self.folder = folder
        self.file_list = []
        with open(metafile) as mf:
            for filename in mf.read().splitlines():
                self.file_list.append(filename)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        id, crop, trash, env_list = self.get_info(filename)
        #print(os.path.join(self.folder, filename))
        img = Image.open(os.path.join(self.folder, filename)) # PIL Image
        img_tensor = ToTensor()(img)
        #trash = torch.FloatTensor(trash)
        #env_list = torch.FloatTensor(env_list)
        return img_tensor, trash, env_list


def mod_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(mod_collate_err_msg_format.format(elem.dtype))

            return mod_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: mod_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(mod_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        #if not all(len(elem) == elem_size for elem in it):
            #raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [mod_collate(samples) for samples in transposed]

    raise TypeError(mod_collate_err_msg_format.format(elem_type))



