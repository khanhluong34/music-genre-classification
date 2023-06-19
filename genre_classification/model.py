from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch


class BERT_BiLSTM(nn.Module):
    def __init__(self):
        super(BERT_BiLSTM, self).__init__()
        
        self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=256, batch_first=True)
        self.fc = nn.Linear(512, 6)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        
        )
        sequence_output = bert_outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        logits = self.fc(lstm_output[:, -1])
        logits = F.softmax(logits, dim=1)
        return logits
    
class LinearClassifier(nn.Module):
    def __init_(self):
        super(LinearClassifier, self).__init__()
        self.dropout1 = nn.Dropout(0.3)
        self.linear1= nn.Linear(768, 128)
        self.relu = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.3)
        self.linear = nn.Linear(128, 6)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x
        
        

class BERTClassification(nn.Module):
    def __init__(self):
        super(BERTClassification, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.linear_classifier = LinearClassifier()

    
    def forward(self, input_ids, attn_mask, token_type_ids):
        outputs = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        logits = self.linear_classifier(pooled_output)

        return logits
