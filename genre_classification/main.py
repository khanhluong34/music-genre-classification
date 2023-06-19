from model import BERT_BiLSTM, BERTClassification, DistilBERTClassification
from train import run_train 
from utils import get_dataloader
from transformers import DistilBertTokenizer
import torch
import os

if __name__ == '__main__':
    
    data_path = 'data/data.csv'
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader, valid_loader, test_loader = get_dataloader(data_path, tokenizer=tokenizer)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DistilBERTClassification()
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