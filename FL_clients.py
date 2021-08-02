from torch.utils.data import DataLoader
import torchvision
from FL_data import DataSplit
import torch

class Client():
    def __init__(self, id, localTrainSet, device, modelPoison) -> None:
        self.device = device
        self.id = id
        self.updateNum = 0
        self.localTrainSet = localTrainSet
        self.trainLoader = None
        self.localParams = None
        self.modelPoison = modelPoison

        #模型投毒，改变数据标签,目标将 0 识别为 8
        if self.modelPoison:
            localTrainSet_Poisoned = []
            for item in self.localTrainSet:
                if item[1] == 0:
                    localTrainSet_Poisoned.append((item[0], 8))
                else:
                    localTrainSet_Poisoned.append(item)
            self.localTrainSet = localTrainSet_Poisoned
            del localTrainSet_Poisoned


    def localUpdate(self, Epoch, batchsize, Net, lossFun, optimizer, globalParams):
        print('# client{} 开始进行本地更新 #'.format(self.id))
        if self.modelPoison:
            print('# client{} 为恶意端 #'.format(self.id))
        self.trainLoader = DataLoader(self.localTrainSet, batch_size=batchsize, shuffle=True, num_workers=0)
        Net.load_state_dict(globalParams, strict=True)
        self.updateNum += 1
        for epoch in range(Epoch):
            running_loss = 0.0
            num = 0
            for i, data in enumerate(self.trainLoader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                optimizer.zero_grad()
                outputs = Net(inputs)
                loss = lossFun(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                num += 1
            print('client%d epoch%d loss: %.4f' % (self.id, epoch+1, running_loss/num))
        
        #根据缩放因子反向放大上传的模型参数
        if self.modelPoison and self.id == 1:
            poisonParams = {}
            for key, var in Net.state_dict().items():
                poisonParams[key] = 5*(var-globalParams[key]) + globalParams[key]
            print('# 返回恶意模型参数 #')
            return poisonParams

        return Net.state_dict()


class ClientsManager():
    def __init__(self, dataSetName, isIID, clientsNum, device=None, modelPoison=False) -> None:
        self.dataSetName = dataSetName
        self.isIID = isIID
        self.clientsNum = clientsNum
        self.clients = {}
        self.testLoader = None
        self.modelPoison = modelPoison
        
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #初始化所有的本地client
        self.clientsInit()

    def clientsInit(self):
        dataSplit = DataSplit(self.dataSetName, self.isIID, self.clientsNum)
        self.testLoader = DataLoader(dataSplit.testset, batch_size=10, shuffle=True, num_workers=0)
        for i in range(self.clientsNum):
            client = Client( i+1, dataSplit.getTrainSet(i+1), self.device, (self.modelPoison if i==0 else False))
            self.clients['client{}'.format(i+1)] = client

if __name__ == '__main__':
    
    class Net(torch.nn.Module):
        def __init__(self) :
            super(Net, self).__init__()
            self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(12, 24, 3, 1, 1),
                                            torch.nn.ReLU(),
                                            torch.nn.MaxPool2d(2, 2)) 
            self.dense = torch.nn.Sequential(torch.nn.Linear(16*16*24, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(p = 0.5),
                                            torch.nn.Linear(256, 10))
            
        def forward(self, x) :
            x = self.conv1(x)
            x = torch.flatten(x, 1)
            x = self.dense(x)
            return x

    CM = ClientsManager('cifar10', True, 10)
    print(CM.clients)
    net = Net()
    net.cuda()

    lossFun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    CM.clients['client1'].localUpdate(10, 10, net, lossFun, optimizer,net.state_dict())