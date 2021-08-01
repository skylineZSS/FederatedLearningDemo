import torchvision
import torchvision.transforms as transforms
import random


class DataSplit():
    def __init__(self, dataSetName, isIID, clientsNum) -> None:
        self.dataSetName = dataSetName
        self.isIID = isIID
        self.clientsNum = clientsNum
        self.trainset = None
        self.testset = None
        

        self.prepareData()
        

    def prepareData(self):
        if self.dataSetName == 'mnist':
            # transform = transforms.Compose([
            #     transforms.Resize((224, 224)),
            #     transforms.Grayscale(3),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
            # ])
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
            self.trainset = list(torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform))
            self.testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            print('# mnist数据加载成功 #')
            if not self.isIID:
                self.trainset.sort(key=lambda x:x[1])
            else:
                random.shuffle(self.trainset)


        elif self.dataSetName == 'cifar10':
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
            self.trainset = list(torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform))
            self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            print('# cifar10数据加载成功 #')
            if not self.isIID:
                self.trainset.sort(key=lambda x:x[1])
            else:
                random.shuffle(self.trainset)

        else:
            raise ValueError("### dataSetName must be 'mnist' or 'cifar10' ###")
        print('训练数据总数： %d' % (len(self.trainset)))
        print('测试数据综述： %d' % (len(self.testset)))
        # count = {str(x):0 for x in range(10)}
        # for i in range(len(self.trainset)):
        #     count[str(self.trainset[i][1])] += 1
        # for num, total in count.items():
        #     print('%s : %d' % (num, total))

    def getTrainSet(self, clientNum):
        if clientNum>self.clientsNum:
            raise ValueError('### clientNum必须小于等于clientsNum ###')
        #将所有训练数据分成2*clientsNum的片段，每个client得到2个片段
        fractionNum = int(len(self.trainset)/(2*self.clientsNum))
        clientData = self.trainset[(clientNum-1)*fractionNum:clientNum*fractionNum] + self.trainset[(clientNum-1+self.clientsNum)*fractionNum:(clientNum+self.clientsNum)*fractionNum]
        return clientData

