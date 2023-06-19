import pandas as pd
from datasets import GenreDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import numpy as np

BATCH_SIZE = 16
NUM_WORKERS = 2

def get_dataset(data_path, tokenizer):
    data = pd.read_csv(data_path)
    # concatenate song name and song lyric
    data['lyric'] = data['song name'] + " " + data['song lyric']
    # convert genre to numeric
    mapping = {'COUNTRY': 0,
               'HIPHOP': 1,
               'INDIE': 2,
               'JAZZ': 3,
               'POP': 4,
               'ROCK': 5}
    data['genre'] = data['category'].map(mapping)
    
    # convert column to numpy array
    lyric = data['lyric']
    genre = data['genre']
    
    # change
    # split data into train, val, test
    x_train, x_test, y_train, y_test = train_test_split(lyric, genre, test_size=0.056, random_state=43)
    # split train into train and val
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.06, random_state=43)
    
    train_df = pd.concat([pd.DataFrame({"lyric": x_train}), y_train], axis=1)
    train_df.reset_index(inplace = True)
    
    val_df = pd.concat([pd.DataFrame({"lyric": x_val}), y_val], axis=1)
    val_df.reset_index(inplace = True)
    
    test_df = pd.concat([pd.DataFrame({"lyric": x_test}), y_test], axis=1)
    val_df.reset_index(inplace = True)
    
    train_dataset = GenreDataset(train_df['lyric'], train_df['genre'], tokenizer=tokenizer)
    valid_dataset = GenreDataset(val_df['lyric'], val_df['genre'], tokenizer=tokenizer)
    test_dataset = GenreDataset(test_df['lyric'], test_df['genre'], tokenizer=tokenizer)
    
    return train_dataset, valid_dataset, test_dataset

def get_dataloader(data_path, tokenizer, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    
    train_dataset, valid_dataset, test_dataset = get_dataset(data_path, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=True,
                                  num_workers=num_workers)
    
    valid_dataloader = DataLoader(valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False,
                                  num_workers=num_workers)
    
    test_dataloader = DataLoader(test_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False,
                                  num_workers=num_workers)
    return train_dataloader, valid_dataloader, test_dataloader
    