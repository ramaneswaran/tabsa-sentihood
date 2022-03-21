<h1 align="center">
  <br />
  Task Based Sentiment Analysis - Sentihood
</h1>

This repo contains TABSA models for Sentihood dataset. The models implemented here use the auxiliary sentence approach introduced in [Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://arxiv.org/abs/1903.09588v1)

Note: The model predictions are present in predictions directory with filename output.jsonl

## Table of Contents

* [Getting started](#getting-started)
* [Main results](#main-results)
	* [Auxiliary Sentences QA M](#auxiliary-sentences-qa-m)
	* [Auxiliary Sentences NLI M](#auxiliary-sentences-nli-m)
* [Training Logs](#training-logs)
* [Acknowledgments](#acknowledgements)

## Getting started

### Dataset
Download the Sentihood dataset and place it in the data directory. 
run `python3 generate_datasets.py` to generate the auxiliary sentence dataset. 

### Usage 
This codebase uses Hydra for configuration management. You can change the configuration present in conf/config.yaml.

Use dataset argument to use either the NLI dataset or QA dataset. 

To use QA_M dataset run 
```
python3 main.py dataset=QA_M
```
or to use NLI_M dataset run 
```
python3 main.py dataset=NLI_M
```


## Main results

#### Auxiliary Sentences QA M
|     **Model**    | **Sent Acc** | **Sent AUC** | **Asp Acc** | **Asp F1** | **Asp AUC** |
|:----------------:|:------------:|:------------:|:-----------:|:----------:|:-----------:|
| BERT             |    0.8964    |    0.9526    |    0.7499   |   0.8558   |    0.9654   |
| RoBERTa          |    0.9243    |    0.9645    |    0.7728   |   0.8509   |    0.9686   |
| RoBERTa 4 layers |    0.9276    |    0.9757    |    0.7419   |   0.8829   |    0.9667   |
| RoBERTA+BiLSTM   |    0.9104    |    0.9645    |    0.7328   |   0.8678   |    0.9591   |

#### Auxiliary Sentences NLI M

|     **Model**    | **Sent Acc** | **Sent AUC** | **Asp Acc** | **Asp F1** | **Asp AUC** |
|:----------------:|:------------:|:------------:|:-----------:|:----------:|:-----------:|
| BERT             |    0.9013    |    0.9525    |    0.7312   |   0.8595   |    0.9631   |
| RoBERTa          |    0.9285    |    0.9665    |    0.7387   |   0.8877   |    0.9654   |
| RoBERTa 4 layers |    0.9219    |    0.9589    |    0.6998   |   0.8234   |    0.9575   |
| RoBERTA+BiLSTM   |    0.9021    |    0.9608    |    0.7344   |   0.9019   |    0.9495   |

### Training logs

Wandb training logs for experiments on QA M and NLI M data. 
* [Wandb Sentihood NLI M Logs](https://wandb.ai/ramaneswaran/Sentihood-NLI)
* [Wandb SentihoodQA M Logs](https://wandb.ai/ramaneswaran/Sentihood)

### Acknowledgements

The evaluation metrics are from [BERT for ABSA](https://github.com/LorenzoAgnolucci/BERT_for_ABSA)
