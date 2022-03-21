import os
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from dataset import SentihoodDataModule
from models import (
    BertBiLSTMModel,
    BertV2Model,
    BertModel
)


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    data_module = SentihoodDataModule(cfg)
    
    if cfg.model.type == 'BertBiLSTMModel':
        model = BertBiLSTMModel(cfg)
    elif cfg.model.type == 'BertV2Model':
        model = BertV2Model(cfg)
    elif cfg.model.type == 'BertModel':
        model = BertModel(cfg)
    else:
        print(f"{cfg.model.type} is invalid model type")
        return
    
    save_dir = hydra.utils.to_absolute_path(cfg.checkpoint.dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    checkpoint_callback = ModelCheckpoint(
         dirpath = save_dir,
         filename = cfg.wandb.name+'-{epoch}-{val_loss:.4f}',
         monitor = 'val_loss')

    wandb_logger = WandbLogger(project=cfg.wandb.project,name=cfg.wandb.name)

    trainer = pl.Trainer(max_epochs=cfg.trainer.epochs, gpus=cfg.trainer.gpus,
                         logger=wandb_logger, callbacks=[checkpoint_callback, EarlyStopping(monitor='val_loss')])

    
    trainer.test(dataloaders=data_module, ckpt_path="best") 
        
if __name__ == "__main__":
    run()
    