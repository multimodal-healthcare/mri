import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torch import nn
from pathlib import Path
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd


class MRITestModel:

    def __init__(self, model_path, mri_dir):
        self.model = torch.jit.load(model_path)
        self.mri_dir = mri_dir

    def predict(self, patient_id):
        mri_paths = list(self.mri_dir.glob("*.png"))
        
        return f'patient has risk of stroke'

