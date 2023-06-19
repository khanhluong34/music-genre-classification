import transformers
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F

# define tokenizer 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
NUM_CLASSES = 6

class GenreDataset(Dataset):
    def __init__(self, lyrics, labels, tokenizer=tokenizer, max_length = 256, num_classes=NUM_CLASSES):
        self.lyrics = lyrics
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        raw_lyric = self.lyrics[idx]
        label = self.labels[idx]
        
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        
        encoded_lyric = self.tokenizer(raw_lyric, 
                                       add_special_tokens=True,
                                       max_length=self.max_length, 
                                       padding='max_length', 
                                       truncation=True, 
                                       return_tensors='pt',
                                       return_token_type_ids=True,
                                       return_attention_mask=True)
        lyric = {
            'input_ids': encoded_lyric['input_ids'].flatten(),
            'attention_mask': encoded_lyric['attention_mask'].flatten(),
            'token_type_ids': encoded_lyric['token_type_ids'].flatten(),
        }
        
        return lyric, label