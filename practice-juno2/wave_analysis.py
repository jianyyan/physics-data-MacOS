#################################
# TODO:
# 分段读取波形，计算各个波形对应的光子数
# 输出预处理后的数据
#################################
import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import argparse
# 定义常数
EventID_num=2000


# 读取标准输入

'''parser = argparse.ArgumentParser()
parser.add_argument("-w", "--waveform", dest="wav", type=str, help="Waveform file")
parser.add_argument("-o", "--output", dest="opt", type=str, help="Output file")
args = parser.parse_args()'''

waveform_filename = 'data/Training_set/final-2.h5'
output_filename = 'data/wave_analysis/output.h5'
# 定义函数
# 功能： 读入(n,1000)数组，计算各图的PE_number,输出(n,)数组
def Waveform_analysis():
    pass
# 按照EventID分段读取输入并进行处理
WaveForm = h5.File(waveform_filename)
output=h5.File(output_filename,'w')
# 得到EventID的个数
EventID_data = WaveForm['Waveform']['EventID']
ChannelID_data = WaveForm['Waveform']['ChannelID']
print(EventID_data)
lastID = 0
event_num = np.zeros(EventID_num+1)
for i in range(len(EventID)):
    if EventID_data[i] != lastID:
        event_num[lastID+1] = i
        lastID += 1
        last_i = i
# 依据EventID的个数循环读入，并依此处理波形
# 动态表格输出
PETruth = np.empty((len(EventID_data),2))
PETruth[:,0] = 
for i in range(EventID_num):
    Event_i_Waveform  = np.stack((ChannelID_data[event_num[i]:event_num[i+1]],WaveForm['Waveform']['Waveform'][event_num[i]:event_num[i+1]]))
    if i == 0:
        output.creat_dataset("PETruth",data = Event_i_Waveform,compression="gzip",chunks=True,maxshape=(None,))
    else:
        output["PETruth"].resize((int(output['PETruth'].shape[0] + event_num[i+1]-event_num[i])),axis=0)
        output["PETruth"][Event_num[i]:] = Event_i_Waveform
WaveForm.close()
output.close()
