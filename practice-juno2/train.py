import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import models 
import torch.nn as nn
import argparse
from trainingset import WaveSet
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default='cpu', help="Input feature")
parser.add_argument("--infeature", type=int, default=1000, help="Input feature")
parser.add_argument("--nhid", type=int, default=128, help="Hidden layer Dimension")
parser.add_argument("--outfeature", type=int, default=1, help="Output feature")
parser.add_argument("--dropout_ratio", type=float, default=0.5, help="Dropout Ratio")
parser.add_argument("--raw_path", type=str, default='data/Training_set', help="Output feature")
parser.add_argument("--batch_size", type=int, default=1, help="training model")
parser.add_argument("--epochs", type=int, default=20, help="Training time")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight Decay to Avoid Overfitting")
parser.add_argument("--in_channel", type=int, default=17612, help="Weight Decay to Avoid Overfitting")
parser.add_argument("--model", type=str, default='MLP', help="training model")
args = parser.parse_args()
print(args)
def build_acc():
    def resolution( output_E, true_E ):
        return torch.sqrt(torch.mean((output_E - true_E)**2/(true_E)))
    return resolution

def test(model,loader):
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
for epoch in range(0,args.epochs):
    outs = []
    loss_list = []
    for i,data in enumerate(train_loader):
        #data[0] = data[0].to(args.device)
        out = model(data[0])
        loss = criterion(out,data[1])
        optimizer.zero_grad()
        # 反向传播计算参数的梯度
        loss.backward()  
        # 使用优化方法进行梯度更新
        optimizer.step()  
        with torch.no_grad():
            outs.append(out.item())
            loss_list.append(loss.item())
    print('train loss:{:4f}\ttrain acc:{:4f}'.format(np.mean(loss_list), np.mean(outs)))
    test(model,test_loader)

