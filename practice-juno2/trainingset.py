import torch
from torch.utils.data import Dataset

class WaveSet(Dataset):
    def __init__(self,args):
        #super(WaveSet,self).__init__()
        self.root = args.raw_path
        self.len = 100
        self.data = torch.rand((self.len,1000))
        #return self
    def __getitem__(self,idx):
        return self.data[idx], torch.rand(1)
    def __len__(self):
        return self.len