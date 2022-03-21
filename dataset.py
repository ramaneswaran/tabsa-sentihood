import hydra

import numpy as np
import pandas as pd

from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer

class AuxiliarySentenceDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
  
    def __getitem__(self, idx):

        # Process the image
        original_sentence = self.data.iloc[idx]['original_sentence']
        auxiliary_sentence = self.data.iloc[idx]['auxiliary_sentence']

        encoded_inputs = self.tokenizer(original_sentence, 
                                        auxiliary_sentence,
                                        padding='max_length', 
                                 max_length=180, truncation=True,
                                 return_tensors='pt')


        # Process the labels
        target = self.data.iloc[idx]['label_id']
        
        # Sentence length
        length = torch.sum(encoded_inputs['attention_mask'].squeeze(0) == 1).long()

        return {
            'input_ids': encoded_inputs['input_ids'].squeeze(),
            'attention_mask': encoded_inputs['attention_mask'].squeeze(0),
            'target':target,
            'length': length,
        }
    
    
    
class SentihoodDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        
        self.train_path = hydra.utils.to_absolute_path(cfg.dataset.train_path)
        self.val_path = hydra.utils.to_absolute_path(cfg.dataset.val_path)
        self.test_path = hydra.utils.to_absolute_path(cfg.dataset.test_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
        
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        
        train_df = pd.read_csv(self.train_path, sep='\t')
        val_df = pd.read_csv(self.val_path, sep='\t')
        test_df = pd.read_csv(self.test_path, sep='\t')
        
        train_df.dropna(inplace=True)
        val_df.dropna(inplace=True)
        test_df.dropna(inplace=True)
        
        self.train_data = AuxiliarySentenceDataset(train_df, self.tokenizer) 
                                      
        self.val_data = AuxiliarySentenceDataset(val_df, self.tokenizer)
        
        self.test_data = AuxiliarySentenceDataset(test_df, self.tokenizer)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.dataloader.train_bs)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.cfg.dataloader.val_bs)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.cfg.dataloader.test_bs)

    def teardown(self, stage: Optional[str] = None):
        # Used to clean-up when the run is finished
        pass