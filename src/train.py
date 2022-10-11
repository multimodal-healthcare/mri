import csv
from tkinter import PROJECTING
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn
import pytorch_lightning as pl

import ast
from pathlib import Path
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from skimage.io import imread

# project_path = Path.cwd().parent
project_path = Path(__file__).parent.parent
mri_dir = project_path/'data/processed/mri'
csv_dir = project_path/'data/processed/csv'


class MRIDataset(Dataset):

    def __init__(self, mri_dir: Path, csv_dir: Path, mode='train') -> None:
        if mode == 'train':
            self.data_dir = mri_dir/'train'
            self.df = pd.read_csv(csv_dir/'train_mri_patients.csv')
        elif mode == 'val':
            self.data_dir = mri_dir/'val'
            self.df = pd.read_csv(csv_dir/'val_mri_patients.csv')
        else:
            raise ValueError('mode should be train, test or val.')
        
        self.mode = mode
        self.mri_png_paths = list(self.data_dir.glob('*.png'))  # list of mri slices as png images
        self.img_size = 128

    def __getitem__(self, idx):
        """
        Return torch Tensor of an MRI slice and its label
        """
        png_path = self.mri_png_paths[idx]        
        data = read_image(str(png_path))
        data = data.type(torch.FloatTensor) 
        data = transforms.Resize((self.img_size, self.img_size))(data)
        patient_id = png_path.name.split('_')[0]
        label = self.df[self.df['patient']==patient_id]['label'].values[0]
        label = ast.literal_eval(label)
        label = torch.Tensor(label)

        return data, label

    def __len__(self):
        return len(self.mri_png_paths)


class MRIModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7),  # change input channel to be 1 instead of 3 
                                      stride=(2, 2), padding=(3, 3), bias=False)
        # add a linear layer at the end for transfer learning
        self.linear = nn.Linear(in_features=self.resnet.fc.out_features,
                                out_features=5)

    # optionally, define a forward method
    def forward(self, xs):
        y_hats = self.resnet(xs)
        y_hats = self.linear(y_hats)
        return y_hats  # we like to just call the model's forward method
    
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        y_hats = self.forward(xs)
        loss = F.binary_cross_entropy_with_logits(y_hats, ys)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        y_hats = self.forward(xs)
        loss = F.binary_cross_entropy_with_logits(y_hats, ys)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


if __name__ == '__main__':
    # hyperparams
    train_batch_size = 200
    val_batch_size = 10
    learning_rate = 1e-3

    train_dataset = MRIDataset(mri_dir, csv_dir, 'train')
    val_dataset = MRIDataset(mri_dir, csv_dir, 'val')
    train_loader = DataLoader(train_dataset, 
                              batch_size=train_batch_size, 
                              shuffle=True, # images are loaded in random order
                              num_workers=12)
                                                
    val_loader = DataLoader(val_dataset, 
                            batch_size=val_batch_size,
                            num_workers=12)

    model = MRIModel()
    trainer = pl.Trainer(accelerator='gpu', devices=int(torch.cuda.is_available()), max_epochs=2)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

