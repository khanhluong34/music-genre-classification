import transformers
from torch.utils.data import Dataset, DataLoader 
from transformers import AutoTokenizer
import torch

# define tokenizer 
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
NUM_CLASSES = 6

# one hot encode labels
def one_hot(label, num_classes=NUM_CLASSES):
    label_encoded = [1 if i == label - 1 else 0 for i in range(num_classes)]
    return label_encoded

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
        
        raw_lyric = self.lyrics.iloc[idx]
        label = self.labels.iloc[idx]
        
        label = one_hot(label, num_classes=self.num_classes)
        # convert label to float tensor
        label = torch.FloatTensor(label)
        
        encoded_lyric = self.tokenizer(raw_lyric, 
                                       add_special_tokens=True,
                                       max_length=self.max_length, 
                                       padding='max_length', 
                                       truncation=True, 
                                       return_tensors='pt',
                                       return_token_type_ids=True,
                                       return_attention_mask=True)
        data = {
            'input_ids': encoded_lyric['input_ids'].flatten(),
            'attention_mask': encoded_lyric['attention_mask'].flatten(),
            'token_type_ids': encoded_lyric['token_type_ids'].flatten(),
            'target': label
        }
        
        return data