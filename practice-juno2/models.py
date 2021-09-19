#################################
# TODO:
# 将预处理数据作为输入
# 搭建**神经网络**计算每个EventID对应的动量值
#################################
#import h5py as h5
import torch
import torch.nn as nn
# 构建 类

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn((out_features, in_features)))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
    def foward(self,x):
        x = torch.matmul(x,self.weight) - self.bias
        return x
    



class MLP(nn.Module):
    def __init__(self,args):
        super(MLP,self).__init__()
        self.infeature = args.infeature
        self.inch = args.inch
        self.emb = args.emb
        self.nhid = args.nhid
        self.outfeature = args.outfeature
        self.dropout_ratio = args.dropout_ratio

        self.feature = nn.Sequential(
            Linear(self.infeature,self.emb)
        )

        self.netlist = nn.Sequential(
            nn.Linear(self.infeature*self.inch,self.nhid),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.nhid,self.nhid//2),
            nn.ReLU(),
            nn.Dropout(self.dropout_ratio),
            nn.Linear(self.nhid//2,self.outfeature)
        )
    def forward(self,x):
        x = self.netlist(x)
        return x

# 标准输入读取预处理数据

# 学习

# 输出参数结果