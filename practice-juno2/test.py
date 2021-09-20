import torch
from torch.utils.data import Dataset
import numpy as np
import os

idx = 2

# 获取训练数据和真实动量值
waveform = torch.load("./event/data_EventID_{}".format(idx))
mean = torch.mean(waveform[:,1:].float())
channel_IDs = waveform[:,0]

# 由于每个EventID对应的有波形的Channel数不同，故先将波形扩展为全部Channel数量
ext_waveform = torch.ones((17612,1000), dtype=torch.float) * mean
real_momentum = torch.load("./event/data_EventID_{}".format(idx))[idx]
ext_waveform[channel_IDs.long()] = waveform[:,1:].float()


