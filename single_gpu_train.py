import os
# 必须在`import torch`语句之前设置才能生效
# torch.cuda.device_count()=3
# torch.cuda.current_device()=0 但使用2
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4"
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
model = model.to(device)  # 默认会使用第一个GPU
optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = []
# training!

for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs, labels=labels)
    loss = outputs[0]  # 对应模型定义中，模型返回始终是tuple
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
    plt.savefig(join('LossFig', "single.png"))
    plt.close()
t1=time.time()
print(t1-t0)  # 运行时长：56.989067792892456秒
