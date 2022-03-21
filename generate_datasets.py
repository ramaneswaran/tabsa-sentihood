import os
import json

from typing import List

from tqdm.notebook import tqdm

import numpy as np
import pandas as pd


sentihood_aspects = ["general", "price", "safety", "transit location"]
idx2aspect = {idx: aspect for idx, aspect in enumerate(sentihood_aspects)}

labels = ["none", "positive", "negative"]
label2id = {label:idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}

locations = ['location - 1', 'location - 2']

def check_aspect_present(aspect, location, opinions):
    """
    Checks if aspect for target location is present in a list of opinions 
    Returns a boolean yes/no and the sentiment if present
    """
    
    for opinion in opinions:
        
        target_entity = opinion['target_entity'].replace("LOCATION2", "location - 2").replace("LOCATION1", "location - 1")
        if opinion['aspect'] == aspect and target_entity == location:
            return True, opinion['sentiment']
    
    return False, None

def process_sentihood_text(text: str):
    """
    Basic preprocessing for sentihood dataset
    Reformats the location entity
    """
    return text.replace("LOCATION2", "location - 2").replace("LOCATION1", "location - 1")

def make_sentihood_aux_dataset(data_list: List, save_path: str, aux_sent_type: str):
    """
    Generates the QA M and NLI M auxiliary sentence dataset
    data_list (list): The sentihood dataset loaded from its json form
    save_path (str): The path to save dataframe at
    aux_sent_type (str): Type of dataset to create, its either QA_M or NLI_M
    """

    outputs = []
    
    for data in tqdm(data_list):

        text = data['text']
        _id = data['id']

        text = process_sentihood_text(text)

        for location in locations:

            if text.find(location) != -1:

                for aspect in sentihood_aspects:

                    aspect_present, sentiment = check_aspect_present(aspect, location, data['opinions'])
                    
                    if aux_sent_type == 'QA_M':
                        aux_text = f"what do you think of the {aspect} of {location} ?"
                    elif aux_sent_type == 'NLI_M':
                        aux_text = f"{location} - {aspect}"
                    else:
                        raise Exception(f"{aux_sent_type} is invalid auxiliary type")
                    
                    label = sentiment.lower() if aspect_present else "none"

                    outputs.append([_id, text, aux_text, label2id[label], label])
        
    header = ['id', 'original_sentence', 'auxiliary_sentence', 'label_id', 'label']
    df = pd.DataFrame(outputs, columns=header)
    df.to_csv(save_path, index=None, sep='\t')

if __name__ == "__main__":
    data_paths = {
    'train': 'data/sentihood-test.json',
    'val': 'data/sentihood-dev.json',
    'test': 'data/sentihood-test.json'
    }
    aux_sent_types = ['QA_M', 'NLI_M']
    save_dir = 'data'

    for aux_sent_type in aux_sent_types:
        for data_type, _path in data_paths.items():

            with open(_path, 'r') as f:
                data_list = json.load(f)
            
            save_path = os.path.join(save_dir, f"{data_type}_{aux_sent_type}.csv")
            
            make_sentihood_aux_dataset(data_list, save_path, aux_sent_type)