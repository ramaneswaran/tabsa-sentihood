import hydra

import numpy as np
import pandas as pd

from typing import Optional

import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pack_padded_sequence

import pytorch_lightning as pl


from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

from sklearn import metrics

from sentihood_metrics import (
    compute_sentihood_sentiment_classification_metrics,
    compute_sentihood_aspect_strict_accuracy,
    compute_sentihood_aspect_macro_F1,
    compute_sentihood_aspect_macro_AUC
)

class SentihoodLightningModel(pl.LightningModule):

    def __init__(self, cfg):
    
        super().__init__()
        self.lr = cfg.optimizer.lr
        self.epoch_count = -1


    def forward(self, input_ids, attention_mask):
        pass
               

    def _shared_eval(self, batch, batch_idx):
        pass

    def training_step(self, batch, batch_idx):
    
        logits, loss = self._shared_eval(batch, batch_idx)

        with torch.no_grad():
            scores = torch.softmax(logits, axis=1)
            preds = torch.argmax(logits, axis=1)
            
        self.log("train_loss", loss)

        return {
          "loss": loss,
          "scores": scores, 
          "preds": preds,
          "targets":batch['target'],
        }

    def validation_step(self, batch, batch_idx):
        """
        Validation step during model training
        Args:
          batch (dictionary): Dictionary containing texts, seq_lens, labels
          batch_idx (int): Index of batch
        Return:
          dict: 
        """

        logits, loss = self._shared_eval(batch, batch_idx)
        
        with torch.no_grad():
            scores = torch.softmax(logits, axis=1)
            preds = torch.argmax(logits, axis=1)

        self.log("val_loss", loss)
        
        return  {
          "loss": loss,
          "scores": scores,
          "preds": preds,
          "targets":batch['target'],
        }
    
    def test_step(self, batch, batch_idx):
        """
        Validation step during model training
        Args:
          batch (dictionary): Dictionary containing texts, seq_lens, labels
          batch_idx (int): Index of batch
        Return:
          dict: 
        """

        logits, loss = self._shared_eval(batch, batch_idx)
        
        with torch.no_grad():
            scores = torch.softmax(logits, axis=1)
            preds = torch.argmax(logits, axis=1)
        
        return  {
          "loss": loss,
          "scores": scores,
          "preds": preds,
          "targets":batch['target'],
        }

    def training_epoch_end(self, outputs):
        """
        Training epoch end 
        Args:
          outputs (dictionary): Dictionary containing training statistics collected during epoch
        Return:
          dict: 
        """

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        targets = torch.cat([x["targets"] for x in outputs]).tolist()
        preds = torch.cat([x["preds"] for x in outputs]).tolist()
        scores = torch.cat([x["scores"] for x in outputs]).tolist()
        
        sent_macro_AUC, sent_acc = compute_sentihood_sentiment_classification_metrics(targets, scores)
        strict_aspect_acc = compute_sentihood_aspect_strict_accuracy(targets, preds)
        aspect_macro_f1 = compute_sentihood_aspect_macro_F1(targets, preds)
        aspect_macro_auc = compute_sentihood_aspect_macro_AUC(targets, scores)
        
        # Log metrics
        self.log_metrics(sent_macro_AUC, sent_acc, strict_aspect_acc,
                         aspect_macro_f1, aspect_macro_auc, "train")
        
        print(f"\nTraining metrics epoch {self.current_epoch}:")
        print("Loss: {:.4f}".format(avg_loss.item()))
        print("Sentiment cccuracy: {:.4f}".format(sent_acc))
        print("Sentiment macro AUC: {:.4f}".format(sent_macro_AUC))
        print("Aspect accuracy: {:.4f}".format(strict_aspect_acc))
        print("Aspect macro F1: {:.4f}".format(aspect_macro_f1))
        print("Aspect macro auc: {:.4f}".format(aspect_macro_auc))
        return None

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end 
        Args:
          outputs (dictionary): Dictionary containing training statistics collected during epoch
        Return:
          dict: 
        """
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        targets = torch.cat([x["targets"] for x in outputs]).tolist()
        preds = torch.cat([x["preds"] for x in outputs]).tolist()
        scores = torch.cat([x["scores"] for x in outputs]).tolist()
        
        if self.epoch_count > -1:
        

            sent_macro_AUC, sent_acc = compute_sentihood_sentiment_classification_metrics(targets, scores)
            strict_aspect_acc = compute_sentihood_aspect_strict_accuracy(targets, preds)
            aspect_macro_f1 = compute_sentihood_aspect_macro_F1(targets, preds)
            aspect_macro_auc = compute_sentihood_aspect_macro_AUC(targets, scores)

            # Log metrics
            self.log_metrics(sent_macro_AUC, sent_acc, strict_aspect_acc,
                             aspect_macro_f1, aspect_macro_auc, "train")
        

        
        
            print(f"\nValidation metrics epoch {self.current_epoch}:")
            print("Loss: {:.4f}".format(avg_loss.item()))
            print("Sentiment cccuracy: {:.4f}".format(sent_acc))
            print("Sentiment macro AUC: {:.4f}".format(sent_macro_AUC))
            print("Aspect accuracy: {:.4f}".format(strict_aspect_acc))
            print("Aspect macro F1: {:.4f}".format(aspect_macro_f1))
            print("Aspect macro auc: {:.4f}".format(aspect_macro_auc))
        else:
            print("Validation sanity fit complete")

        self.epoch_count += 1
        print("-" * 50)

        return None

    def test_epoch_end(self, outputs):
        """
        Training epoch end 
        Args:
          outputs (dictionary): Dictionary containing training statistics collected during epoch
        Return:
          dict: 
        """

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        targets = torch.cat([x["targets"] for x in outputs]).tolist()
        preds = torch.cat([x["preds"] for x in outputs]).tolist()
        scores = torch.cat([x["scores"] for x in outputs]).tolist()
        
        sent_macro_AUC, sent_acc = compute_sentihood_sentiment_classification_metrics(targets, scores)
        strict_aspect_acc = compute_sentihood_aspect_strict_accuracy(targets, preds)
        aspect_macro_f1 = compute_sentihood_aspect_macro_F1(targets, preds)
        aspect_macro_auc = compute_sentihood_aspect_macro_AUC(targets, scores)
        
        # Log metrics
        self.log_metrics(sent_macro_AUC, sent_acc, strict_aspect_acc,
                         aspect_macro_f1, aspect_macro_auc, "test")
        
        print(f"\nTest metrics {self.current_epoch}:")
        print("Loss: {:.4f}".format(avg_loss.item()))
        print("Sentiment cccuracy: {:.4f}".format(sent_acc))
        print("Sentiment macro AUC: {:.4f}".format(sent_macro_AUC))
        print("Aspect accuracy: {:.4f}".format(strict_aspect_acc))
        print("Aspect macro F1: {:.4f}".format(aspect_macro_f1))
        print("Aspect macro auc: {:.4f}".format(aspect_macro_auc))
        return None
    
    def log_metrics(self, sent_macro_AUC, sent_acc, strict_aspect_acc,
                         aspect_macro_f1, aspect_macro_auc, phase):

        """
        Logs metrics
        """
        
        self.log(f"{phase}_sent_macro_auc", sent_macro_AUC)
        self.log(f"{phase}_sent_acc", sent_acc)
        self.log(f"{phase}_aspect_acc", strict_aspect_acc)
        self.log(f"{phase}_aspect_f1",  aspect_macro_f1)
        self.log(f"{phase}_aspect_auc",  aspect_macro_auc)
        
        

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
    
class BertBiLSTMModel(SentihoodLightningModel):
    """
    Bert+BiLSTM model. The BiLSTM takes as input the last hidden state of BERT
    and classification logits are derived from BiLSTM's hidden state
    """

    def __init__(self, cfg):
    
        super().__init__(cfg)
        
        self.model = AutoModel.from_pretrained(cfg.model.name)
        
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        
        hidden_size = self.model.config.hidden_size
    
        self.lstm = nn.LSTM(hidden_size,hidden_size,bidirectional=True)

        
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, cfg.model.num_labels),
          )
        
        if cfg.model.finetune_encoder is False:
            for param in self.model.parameters():
                param.requires_grad = False            
        
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask, length):

        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        

        last_hidden_state = self.dropout(outputs.last_hidden_state.permute(1, 0, 2))
        enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(
            last_hidden_state, length.cpu(), enforce_sorted=False))
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = F.dropout(output_hidden,0.2)  
        
        logits = self.classifier(output_hidden)

        return logits
               

    def _shared_eval(self, batch, batch_idx):
    
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']
        length = batch['length']
        
        logits = self(input_ids, attention_mask, length)
        
        loss = self.criterion(logits, target.long())
        
        return logits, loss
    
class BertV2Model(SentihoodLightningModel):
    """
    An extended Bert model . We use the [CLS] token embedding from the last 4 layers, 
    this is concatenated to provide the hidden sentence representation
    """
    
    def __init__(self, cfg):
    
        super().__init__(cfg)
        
        self.model = AutoModel.from_pretrained(cfg.model.name)
        
        self.dropout = nn.Dropout(self.model.config.hidden_dropout_prob)
        
        hidden_size = self.model.config.hidden_size * 4
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.SiLU(),
            nn.Linear(hidden_size // 4, cfg.model.num_labels),
          )
        
        if cfg.model.finetune_encoder is False:
            for param in self.model.parameters():
                param.requires_grad = False           
        
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, input_ids, attention_mask):

        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        
        pooled_output = torch.cat(tuple([outputs.hidden_states[i][:, 0, :] for i in [-4, -3, -2, -1]]), dim=-1)
        
        logits = self.classifier(self.dropout(pooled_output))

        return logits
               

    def _shared_eval(self, batch, batch_idx):
    
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']
        
        logits = self(input_ids, attention_mask)
        
        loss = self.criterion(logits, target.long())
        
        return logits, loss
    
class BertModel(SentihoodLightningModel):
    """
    Bert model for TABSA. The final pooled output is used for classification
    """
    
    def __init__(self, cfg):
    
        super().__init__(cfg)
        
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model.name, 
                                                            num_labels=cfg.model.num_labels)




    def forward(self, input_ids, attention_mask, labels):

        outputs = self.model(input_ids, attention_mask, labels=labels)

        return outputs.logits, outputs.loss
               

    def _shared_eval(self, batch, batch_idx):
    
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target = batch['target']
        
        logits, loss = self(input_ids, attention_mask, target)
        
        return logits, loss