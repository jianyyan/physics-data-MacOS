import torch
from torch.utils.data import Dataset
import os
import numpy as np

class WaveSet(Dataset):
    def __init__(self,args):
        #super(WaveSet,self).__init__()
        self.root = args.raw_path
        self.len = 17612
        #return self
    def __getitem__(self,idx):
        # 获取训练数据和真实动量值
        waveform = torch.load(os.path.join(self.root, 'data_EventID_{}'.format(idx)))
        mean = torch.mean(waveform[:,1:].float())
        channel_IDs = waveform[:,0]
        
        # 由于每个EventID对应的有波形的Channel数不同，故先将波形扩展为全部Channel数量
        ext_waveform = torch.zeros((17612,1000), dtype=torch.float)
        real_momentum = torch.load(os.path.join(self.root, 'real_momentum'))[idx]
        ext_waveform[channel_IDs.long()] = waveform[:,1:].float() - mean

        return ext_waveform, real_momentum.float()
    def __len__(self):
        return 1999