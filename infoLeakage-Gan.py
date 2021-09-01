from pickle import TRUE
import torch
from torch import cuda
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

def to_img(x):
    out = 0.5*(x +1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

#定义判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.dis(x)
        return x

class discriminator_CNN(nn.Module):
    def __init__(self):
        super(discriminator_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),#32,28,28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)#32,14,14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),#64,14,14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)#64,7,7
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)#这里x是一个batchsize，所以要从第2维开始展开
        x = self.fc(x)
        return x

#定义生成网络
class generator(nn.Module):
    def __init__(self, input_size):
        super(generator, self).__init__() 
        self.gen = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.gen(x)
        return x

class generator_CNN(nn.Module):
    def __init__(self, input_size):
        super(generator_CNN, self).__init__()
        self.fc = nn.Linear(input_size, 3136)
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

#定义联邦学习的模型
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
                                            torch.nn.Linear(256, 2),
                                            torch.nn.Softmax(dim=1))
            
        def forward(self, x) :
            x = self.conv1(x)
            x = torch.flatten(x, 1)
            x = self.dense(x)
            return x

#定义客户端
# class Client():
#     def __init__(self, isAdversary) -> None:
#         self.isAdversary = isAdversary


batch_size = 6
num_epoch = 2
z_dimension = 100
rounds = 100


#训练数据加载
transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307), (0.3081))
                transforms.Normalize((0.5), (0.5))
            ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testLoader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

#为善良/恶意客户端分配数据
trainset_benign = list(filter(lambda x:x[1]==0 or x[1]==1, list(trainset)))#正常客户端训练3，1
trainset_malicious = list(filter(lambda x:x[1]==1, list(trainset)))#恶意客户端只有1的数据
trainset_benign_Loader = DataLoader(trainset_benign, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
trainset_malicious_Loader = DataLoader(trainset_malicious, batch_size=int(batch_size/2), shuffle=True, num_workers=0, drop_last=True)


# D = discriminator()
# G = generator(z_dimension)
net2 = Net()
G = generator_CNN(z_dimension)
net2.cuda()
G.cuda()


#定义两个客户端
#客户端2中判决器D与本地模型公用
net1 = Net()
net1.cuda()

criterion = nn.BCELoss()
criterion_CEL = nn.CrossEntropyLoss()
net1_optimizer = torch.optim.Adam(net1.parameters(), lr=0.001)
net2_optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

globalParams = {}
for key, val in net1.state_dict().items():
    globalParams[key] = val.clone()

#开始训练
for round in range(rounds):
    print('# 开始第 %d 轮联邦学习 #' % (round))
    #全局模型参数
    sumParams = None

    #训练客户端1
    net1.load_state_dict(globalParams)
    for epoch in range(num_epoch):
        running_loss = 0.0
        num = 0
        for i, data in enumerate(trainset_benign_Loader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            # labels = torch.reshape(labels, [batch_size, 1])
            net1_optimizer.zero_grad()
            outputs = net1(inputs)
            # print(labels.dtype)
            # print(outputs)
            loss = criterion_CEL(outputs, labels)
            loss.backward()
            net1_optimizer.step()
            running_loss += loss.item()
            num += 1
        print('client1 epoch%d loss: %.4f' % (epoch+1, running_loss/num))

    #训练客户端2，恶意
    #先利用GAN生成假数据
    net2.load_state_dict(globalParams)
    running_loss = 0.0
    for i in range(int(6000/batch_size)):
        z = torch.randn(batch_size, z_dimension).cuda()
        fake_img = G(z)
        output = net2(fake_img)
        # real_label = Variable(torch.reshape(torch.zeros(batch_size), [batch_size, 1])).cuda()
        real_label = torch.zeros(batch_size, dtype=torch.long).cuda()
        # print(output)
        # print(real_label)
        g_loss = criterion_CEL(output, real_label)

        #生成器优化
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        running_loss += g_loss.item()
    print('client2 GeneratorNet loss: %.4f' % (running_loss/(50)))

    #生成假数据
    
    # fake_data = [(item, 0) for item in fake_img]
    # trainset_malicious_Loader = DataLoader(trainset_malicious+fake_data, batch_size=batch_size, shuffle=True, num_workers=0)

    #训练客户端2本地模型
    for epoch in range(0):
        running_loss = 0.0
        num = 0
        for i, data in enumerate(trainset_malicious_Loader, 0):
            inputs, labels = data[0].cuda(), data[1].cuda()
            #利用G生成假数据
            z = torch.randn(int(batch_size/2), z_dimension).cuda()
            fake_img = G(z)
            fake_label = torch.zeros(int(batch_size/2), dtype=torch.long).cuda()
            inputs = torch.cat((inputs, fake_img), dim=0)
            labels = torch.cat((labels, fake_label), dim=0)
            # print(labels)

            net2_optimizer.zero_grad()
            outputs = net2(inputs)
            loss = criterion_CEL(outputs, labels)
            loss.backward()
            net2_optimizer.step()
            running_loss += loss.item()
            num += 1
        print('client2 epoch%d loss: %.4f' % (epoch+1, running_loss/num))

    #整合全局模型
    globalParams = net1.state_dict()
    # for key in globalParams:
    #     globalParams[key] = (globalParams[key] + net2.state_dict()[key])/2
    
    z = torch.randn(10, z_dimension).cuda()
    fake_img = G(z)
    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './img/fake_images-{}.png'.format(round+1))

# torch.save(G.state_dict(), './generator.pth')
# torch.save(D.state_dict(), './discriminator.pth')