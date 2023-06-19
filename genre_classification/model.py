from transformers import BertModel
import torch.nn as nn
import torch.nn.functional as F

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
    

