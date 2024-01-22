import torch
from torch.utils.data import Dataset

class SimpleSequenceDKTDataset(Dataset):
    def __init__(self, user_ids, X, mask, y=None, max_length=512, train=True):
        
        super().__init__()
        self.train = train

        self.user_ids = user_ids
        self.X = X
        self.mask = mask
        
        if self.train:
            self.y = y
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        item = {'user_id': self.user_ids[index]}
        item['X'] = self.X[index]
        item['mask'] = self.mask[index]
        if self.train:
            item['y'] = self.y[index]
        return item
    
    def get_user_ids(self):
        return self.user_ids