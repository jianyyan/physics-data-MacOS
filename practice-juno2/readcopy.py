import h5py as h5
import numpy as np
from numpy.lib.function_base import percentile, piecewise
import torch
import time
start = time.time()
# 从训练集中读取波形数据，并存到./event文件夹中，以便之后学习调用
ord_sum=0
for file_ord in range(10,11):
    with h5.File("final-{}.h5".format(file_ord), "r") as ipt:
        # 读取waveform，存到a中
        EventID_data = ipt['Waveform']['EventID']
        print("eve")
        end=time.time()
        print("Running time %s seconds"%(end-start))
        ChannelID_data = ipt['Waveform']['ChannelID']
        end=time.time()
        print("Running time %s seconds"%(end-start))
        eventid_max=np.max(EventID_data)
        eventid_min=np.min(EventID_data)
        EventID_num = eventid_max+1-eventid_min
        event_num = np.zeros(EventID_num+1,dtype=int)
        for i in range(eventid_min, eventid_max+1):
            event_num[i+1]  = len(np.where(EventID_data[:] == i)[0])+event_num[i]
        # 将waveform按照每个EventID进行分类，分别存储为文件，之后作为输入
        print(event_num)
        for i in range(eventid_min, eventid_max+1):
            print("event",i+ord_sum)
            a = ipt['Waveform'][event_num[i]:event_num[i+1]]
            len_info = event_num[i+1]- event_num[i]    
            f = np.array([[wave[0], wave[1]] for wave in a])
            channel_info = np.array([[wave[1]] for wave in a])
            wave_info = np.array([[wave[2]] for wave in a]).reshape(len_info,1000)
            channel_wave_info = np.hstack([channel_info, wave_info])
            torch.save(torch.tensor(channel_wave_info),"./event/data_EventID_{}".format(i+ord_sum))

        end=time.time()
        print("Running time %s seconds"%(end-start))

        ord_sum+=EventID_num