from model import BERT_BiLSTM
from train import run_train 
from utils import get_dataloader
import torch
import torch.nn as nn
import os

if __name__ == '__main__':
    
    data_path = 'data/data.csv'
    train_loader, valid_loader, test_loader = get_dataloader(data_path)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERT_BiLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    epochs = 10
    save_path = '/results'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    run_train(model,
              train_loader,
              valid_loader,
              test_loader,
              criterion,
              optimizer,
              device,
              epochs,
              save_path
              )