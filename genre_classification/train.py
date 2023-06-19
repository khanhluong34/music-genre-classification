from model import BERT_BiLSTM
import torch
import numpy as np 
import os

def run_train(model, train_loader, val_loader, test_loader, criterion, optimizer, device, epochs, save_path):
    print('########################## Training Start ##########################')
    best_acc = 0
    print('\n')
    model.to(device)
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        model, train_acc, train_loss = train_phase(model, train_loader, criterion, optimizer, device) 
        val_acc, val_loss = eval_phase(model, val_loader, criterion, device)
        print(f'Train accuracy: {train_acc:.4f}. Train loss: {train_loss:.4f}, Val accuracy: {val_acc:.4f}. Val loss: {val_loss:.4f}')
        if val_acc > best_acc:
            best_acc = val_acc
            print('The validation accuracy increases, save the best model ...')
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
        
    
    print('########################## End training ##########################')
    print("The best model with validation accuracy: {:.4f}".format(best_acc))
    print("Test the best model on test set ...")
    test_acc = test_phase(model, test_loader, device)
    print("Test accuracy: {:.4f}".format(test_acc))
        
def train_phase(model, train_loader, criterion, optimizer, device):
    train_loss_epoch = []
    correct_predictions = 0
    num_samples = 0
    
    model.train()
    for i, (data, target) in enumerate(train_loader):
        
        input_ids = data['input_ids'].to(device)
        attenttion_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        target = target.to(device)
         
        optimizer.zero_grad()
        logits = model(input_ids, attenttion_mask, token_type_ids)
        loss = criterion(logits, target)
        train_loss_epoch.append(loss.item())
        _, preds = torch.max(logits, dim=1)
        _, targ = torch.max(target, dim=1)
        correct_predictions += torch.sum(preds == targ)
        num_samples += len(preds)
        loss.backward()
        optimizer.step()
        
    return model, float(correct_predictions)/float(num_samples), np.mean(train_loss_epoch)

def eval_phase(model, val_loader, criterion, device):
    model.eval()
    val_loss_epoch = []
    correct_predictions = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            input_ids = data['input_ids'].to(device)
            attenttion_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            target = target.to(device)
            
            logits = model(input_ids, attenttion_mask, token_type_ids)
            loss = criterion(logits, target)
            val_loss_epoch.append(loss.item())
            _, preds = torch.max(logits, dim=1)
            _, targ = torch.max(target, dim=1)
            correct_predictions += torch.sum(preds == targ)
            num_samples += len(preds)
            
    return float(correct_predictions)/float(num_samples), np.mean(val_loss_epoch)

def test_phase(save_path, test_loader, device):
    model = BERT_BiLSTM()
    model.load_state_dict(torch.load(os.path.join(save_path, 'best_model.pt')))
    model.eval()
    model.to(device)
    correct_predictions = 0
    num_samples = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            input_ids = data['input_ids'].to(device)
            attenttion_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            target = target.to(device)
            
            logits = model(input_ids, attenttion_mask, token_type_ids)
            _, preds = torch.max(logits, dim=1)
            _, targ = torch.max(target, dim=1)
            correct_predictions += torch.sum(preds == targ)
            num_samples += len(preds)
    test_acc = float(correct_predictions)/float(num_samples)
    return test_acc        
   