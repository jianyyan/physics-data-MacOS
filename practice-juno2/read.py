import h5py as h5
import numpy as np
from numpy.lib.function_base import percentile, piecewise
import torch

# 从训练集中读取波形数据，并存到./event文件夹中，以便之后学习调用
with h5.File("final-2.h5", "r") as ipt:
    # 读取waveform，存到a中
    a = ipt['Waveform']
    len_info = len(a)

    # 将waveform按照每个EventID进行分类，分别存储为文件，之后作为输入
    f = np.array([[wave[0], wave[1]] for wave in a])
    channel_info = np.array([[wave[1]] for wave in a])
    wave_info = np.array([[wave[2]] for wave in a]).reshape(len_info,1000)
    channel_wave_info = np.hstack([channel_info, wave_info])
    for i in range(np.min(f[:,0]), np.max(f[:,0])+1):
        torch.save(torch.tensor(channel_wave_info[np.where(f[:,0] == i)]),"./event/data_EventID_{}".format(i))

    # 读取particletruth作为标签
    b = ipt['ParticleTruth']
    momentum_info = np.array([truth[4] for truth in b ])
    torch.save(torch.tensor(momentum_info),"./event/real_momentum")