# PyTorch 单机多GPU 训练方法与原理整理

参考[原项目](https://github.com/jia-zhuang/pytorch-multi-gpu-training)，在此基础上，对代码文件新增内容：

1. 运行时长的测试结果

2. torchrun的运行示例（在`ddp_train.py`的注释里）

3. 一些小的更新，例如参数--local_rank的传递方法，一些注释等。

测试gpu型号：a6000

其他具体修改细节请用代码比较器进行比较。


# `single_gpu_train.py`和`data_parallel_train.py`使用方法
直接调python运行此文件，参考原项目。

# `ddp_train.py`使用方法(单个主机4个gpu)

方法1.在命令行里输入：

```
$ python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 ddp_train.py 
```

方法2.在命令行里输入：

```
$ torchrun --standalone --nnodes=1 --nproc-per-node=4 ddp_train.py 
```

其中torchrun来自python库里的torch包：

```
$ whereis torchrun
torchrun: /home/david/anaconda3/envs/torchEnv/bin/torchrun
```

多节点请参考官方文档[ torchrun (Elastic Launch)](https://docs.pytorch.org/docs/stable/elastic/run.html)
