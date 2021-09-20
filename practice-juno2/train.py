import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import models 
import torch.nn as nn
import argparse
from trainingset import WaveSet

# 传递参数部分
parser = argparse.ArgumentParser()
# 设置train.py运行device，可以改为cuda
parser.add_argument("--device", type=str, default='cpu', help="Input feature")
parser.add_argument("--infeature", type=int, default=1000, help="Input feature")
# 设置中间层神经元个数
parser.add_argument("--nhid", type=int, default=128, help="Hidden layer Dimension")
parser.add_argument("--outfeature", type=int, default=1, help="Output feature")
# 训练是随机放弃一部分神经元以提高准确率时，随机放弃的概率，可以在[0,1]之间调整
parser.add_argument("--dropout_ratio", type=float, default=0.5, help="Dropout Ratio")
# 预处理后文件存储的文件夹名称
parser.add_argument("--raw_path", type=str, default='event', help="Output feature")
# 一次放入训练的个数，增大为128/256会更稳定，有GPU会更快
parser.add_argument("--batch_size", type=int, default=1, help="training model")
# 迭代次数，一般到一定值就会收敛
parser.add_argument("--epochs", type=int, default=20, help="Training time")
# 学习率，越大越快，但越不稳定，可以调节为阶梯下降，每跑几个就变为原先的一半
parser.add_argument("--lr", type=float, default=0.0005, help="Learning Rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight Decay to Avoid Overfitting")
parser.add_argument("--in_channel", type=int, default=17612, help="Weight Decay to Avoid Overfitting")
parser.add_argument("--model", type=str, default='MLP', help="training model")
# 将波形变为的feature数
parser.add_argument("--emb", type=int, default=4, help="embedding dimension")
parser.add_argument("--inch", type=int, default=17612, help="number of input channel")

args = parser.parse_args()
print(args)

# loss计算函数
def build_acc():
    def resolution( output_E, true_E ):
        return torch.sqrt(torch.mean((output_E - true_E)**2/(true_E)))
    return resolution

# 测试集进行测试
def test(model,loader):
    model.eval()
    with torch.no_grad():
        outs = []
        loss_list = []
        for i,data in enumerate(loader):
            out = model(data[0])
            loss = criterion(out,data[1])
            outs.append(out.item())
            loss_list.append(loss.item())
        print('test loss:{:4f}\ttest acc:{:4f}'.format(np.mean(loss_list), np.mean(outs)))

# build training set
train_set = WaveSet(args)
test_set = WaveSet(args)

# build model
Model_t = getattr(models,args.model)
model = Model_t(args)
print(model)
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

# build data loader
train_loader = DataLoader(train_set,batch_size=args.batch_size,shuffle=True,drop_last=True)
test_loader = DataLoader(test_set,batch_size=1,shuffle=False,drop_last=True)

# build loss
criterion = nn.MSELoss().to(args.device)
accuracy = build_acc()

# training & testing
for epoch in range(0,args.epochs):
    model.train()
    outs = []
    loss_list = []
    targets = []
    for i,data in enumerate(train_loader):
        data = data.to(args.device)
        out = model(data[0])
        loss = criterion(out,data[1])
        optimizer.zero_grad()
        # 反向传播计算参数的梯度
        loss.backward()  
        # 使用优化方法进行梯度更新
        optimizer.step()  
        with torch.no_grad():
            targets.append(data[1].item())
            outs.append(out.item())
            loss_list.append(loss.item())
    outs = torch.tensor(outs)
    targets = torch.tensor(targets)
    print('train loss:{:4f}\ttrain acc:{:4f}'.format(np.mean(loss_list), accuracy(outs,targets)))
    test(model,test_loader)

# 用训练好的model将结果输出到h5文件

# 获取训练数据和真实动量值
waveform = torch.load('data_EventID_{}'.format(idx)))
mean = torch.mean(waveform[:,1:].float())
channel_IDs = waveform[:,0]

# 由于每个EventID对应的有波形的Channel数不同，故先将波形扩展为全部Channel数量
ext_waveform = torch.zeros((17612,1000), dtype=torch.float)
real_momentum = torch.load(os.path.join(self.root, 'real_momentum'))[idx]
ext_waveform[channel_IDs.long()] = waveform[:,1:].float() - mean