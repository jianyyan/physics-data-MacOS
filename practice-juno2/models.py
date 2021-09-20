#################################
# 将预处理数据作为输入
# 搭建**神经网络**计算每个EventID对应的动量值
#################################
import h5py as h5
import torch
import torch.nn.functional as F
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,args):

        # 参数初始化
        super(MLP,self).__init__()
        self.infeature = args.infeature
        self.inch = args.inch
        self.emb = args.emb
        self.nhid = args.nhid
        self.outfeature = args.outfeature
        self.dropout_ratio = args.dropout_ratio

        # 提取特征
        self.feature = nn.Sequential(
            nn.Linear(self.infeature,self.emb*8,bias=True),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.emb*8,self.emb,bias=True),
            nn.ReLU(),
            nn.Dropout()
        )

        # 回归网络
        self.netlist = nn.Sequential(
            nn.Linear(self.emb*self.inch,self.nhid),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.nhid,self.nhid//8),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.nhid//8,self.outfeature)
        )

    def forward(self,x):
        x = self.feature(x)
        x = self.netlist(x.view(-1,))
        return x