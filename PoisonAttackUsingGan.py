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
        x = x.view(x.size(0), 1, 28, 28)
        return x

batch_size = 6
num_epoch = 20
z_dimension = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#判决器
D = Net()
D.load_state_dict(torch.load('discriminator.pth'))
print(D)
#生成器
G = generator(input_size=100)
D.to(device)
G.to(device)





criterion_CEL = nn.CrossEntropyLoss()
# net1_optimizer = torch.optim.Adam(net1.parameters(), lr=0.001)
# net2_optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)

num = 0
running_loss = 0
for i in range(10000):
    z = torch.randn(batch_size, z_dimension).to(device)
    fake_img = G(z)
    outputs = D(fake_img)
    # print(outputs)
    #目标是生成0
    real_label = torch.zeros(batch_size, dtype=torch.long)
    # for j in range(batch_size):
    #     real_label[j][0] = 1

    loss = criterion_CEL(outputs, real_label)
    
    g_optimizer.zero_grad()
    loss.backward()
    g_optimizer.step()
    running_loss += loss.item()

    if i%1000 == 0:
        print("G loss: %.4f" % (running_loss/1000))
        running_loss = 0

        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(int(i/1000)))
