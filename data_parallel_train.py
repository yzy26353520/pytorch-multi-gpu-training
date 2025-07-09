import os
# 使用cuda:0,1,2,3
# torch.cuda.device_count()=2
# torch.cuda.current_device()=0 从2和3里选，代表2而非0
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Net
from data import train_dataset
import matplotlib.pyplot as plt
import time
t0=time.time()
join=os.path.join

device = torch.device('cuda')
batch_size = 64

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = Net()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
model = nn.DataParallel(model)  # 就在这里wrap一下，模型就会使用所有的GPU
losses = []
# training!
for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs, labels=labels)
    loss = outputs[0]  # 对应模型定义中，模型返回始终是tuple
    loss = loss.mean()  # 将多个GPU返回的loss取平均
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log
    losses.append(loss.detach().cpu().numpy())
    plt.plot(losses)
    plt.title("Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.savefig(join('LossFig', "dp.png"))
    plt.close()
t1=time.time()
print(t1-t0)
# 若干个GPU运行时长：
# 1个：53.07090210914612秒
# 2个：66.04949569702148秒
# 3个：61.15757727622986秒
# 4个：61.876888036727905秒
