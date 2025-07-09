# 方法1.使用torch.distributed.launch运行，可以运行成功，但将被官方弃用：
# python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 ddp_train.py
# 方法2.使用torchrun运行：
# torchrun --standalone --nnodes=1 --nproc-per-node=4 ddp_train.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
# 官方弃用argparse传入--local_rank，而是传入--local-rank，但推荐os.environ['LOCAL_RANK']
local_rank=int(os.environ['LOCAL_RANK'])

import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler

from model import Net
from data import train_dataset, test_dataset
import matplotlib.pyplot as plt
import time
t0=time.time()
join=os.path.join

torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)
torch.distributed.init_process_group(backend='nccl')

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 64

model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training!

train_sampler = DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
losses=[]
for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)[0]
    loss = criterion(outputs, labels)
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
    plt.savefig(join('LossFig', "ddp.png"))
    plt.close()
torch.distributed.destroy_process_group()  # 必须要，否则内存/现存泄露
t1=time.time()
print(t1-t0)
# 1.使用torch.distributed.launch运行，若干GPU时长：
# 1个：55.651301860809326秒
# 2个：33.38007354736328、33.39727568626404秒
# 3个：27.27257013320923、27.338772773742676、27.323105335235596秒
# 4个：22.365924835205078、22.32585620880127、22.327811241149902、22.25014042854309秒

# 2.使用torchrun运行：
# 1个：55.35274004936218秒
# 2个：34.10897397994995、34.05248689651489秒
# 3个：26.43994951248169、26.44611430168152、26.47258949279785秒
# 4个：19.881168365478516、19.992006540298462、20.015733003616333、20.046803951263428秒
