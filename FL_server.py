import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np
from FL_clients import ClientsManager

from torchvision.transforms.transforms import Grayscale

if __name__ == '__main__':
    
    class Net(torch.nn.Module):
        def __init__(self) :
            super(Net, self).__init__()
            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 12, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(12, 24, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(2, 2)) 
            self.dense = torch.nn.Sequential(torch.nn.Linear(14*14*24, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p = 0.5),
                                            torch.nn.Linear(256, 10))
            
        def forward(self, x) :
            x = self.conv1(x)
            x = torch.flatten(x, 1)
            x = self.dense(x)
            return x

    CLIENTS_NUM = 5
    FL_ROUNDS = 5
    LOCAL_EPOCH = 5
    LOCAL_BATCHSIZE = 10
    MODEL_POISON = False



    CM = ClientsManager('mnist', False, CLIENTS_NUM, modelPoison=MODEL_POISON)

    print(CM.clients)
    net = Net()
    net.cuda()

    lossFun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    globalParams = {}
    for key, val in net.state_dict().items():
        globalParams[key] = val.clone()

    #开始进行联邦学习
    for i in range(FL_ROUNDS):
        print('# 开始第 %d 轮联邦学习 #' % (i))
        sumParams = None
        for j in range(CLIENTS_NUM):
            #获取本地client的梯度
            localParams = CM.clients['client{}'.format(j+1)].localUpdate(LOCAL_EPOCH, LOCAL_BATCHSIZE, net, lossFun, optimizer, globalParams)
            if sumParams==None:
                sumParams = {}
                for key, var in localParams.items():
                    sumParams[key] = var.clone()
            else:
                for key in sumParams:
                    sumParams[key] += localParams[key]
        
        for key in globalParams:
            globalParams[key] = sumParams[key]/CLIENTS_NUM
        
        #查看全局模型在测试集上的准确率
        with torch.no_grad():
            correct, total = 0, 0
            res = {x:[0,0] for x in range(10)}
            net.load_state_dict(globalParams, strict=True)
            for data in CM.testLoader:
                inputs, labels = data[0].cuda(), data[1].cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                labels = labels.cpu()
                predicted = predicted.cpu()
                for j in range(len(labels)):
                    res[int(labels[j])][0] += 1            
                    if predicted[j] == labels[j]:
                        res[int(labels[j])][1] += 1
                total += labels.size(0)
                correct += (predicted==labels).sum().item()
            print('# 第 %d 轮学习的全局模型准确率为： %.2f %%' % (i+1, 100*correct/total))
            for key, val in res.items():
                print('# 类别 %d 的识别准确率为： %.2f %%' % (key, 100*val[1]/val[0]))

        #查看模型投毒攻击的目标识别率，将0识别为8的准确率
        with torch.no_grad():
            correct, wrong, total = 0, 0, 0
            net.load_state_dict(globalParams, strict=True)
            for data in CM.testLoader:
                inputs, labels = data[0].cuda(), data[1].cuda()
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                for j in range(len(labels)):
                    if labels[j] == 0:
                        total += 1
                        if predicted[j] == 8:
                            correct += 1
                        else:
                            wrong += 1
            print('# 第 %d 轮模型投毒攻击中 0-》8 的准确率为： %.2f %%' % (i+1, 100*correct/total))
            print('# 第 %d 轮模型投毒攻击中 0-》0 的准确率为： %.2f %%' % (i+1, 100*wrong/total))

    





