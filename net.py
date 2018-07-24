from __future__ import print_function, division
import os
import shutil
import torch
import torchvision as tv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import torch.optim as optim


#读取数据

transform = transforms.Compose([transforms.Resize(204),transforms.CenterCrop(204),transforms.ToTensor(),transforms.Normalize(mean= [0.5,0.5,0.5],std= [0.5,0.5,0.5])])

train_dataset = tv.datasets.ImageFolder(root = 'data/train',transform = transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size = 4,shuffle = True,num_workers = 4)

test_dataset = tv.datasets.ImageFolder(root = 'data/certify',transform = transform)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4,shuffle = True,num_workers = 4)


# 建立网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.conv3 = nn.Conv2d(16,32,5)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*22*22,10000)
        self.fc2 = nn.Linear(10000,1000)
        self.fc3 = nn.Linear(1000,100)
        self.fc4 = nn.Linear(100,2)
        
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,32*22*22)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

net = Net().cuda()


#定义误差和优化函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)

#训练网络

for ehco in range(3):
    running_loss = 0
    for i,data in enumerate(train_loader,0):
        #print('i: ',i)
        inputs,labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        #print(labels)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if i%1000 == 999:
            print('[%d %5d] loss: %0.3f' % (ehco+1,i+1,running_loss/1000))
            running_loss = 0

print('Finished')


#测试数据
total = 0
correct = 0
for i,data in enumerate(test_loader,0):
    inputs,labels = data
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = net(inputs)
    _,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy of the network on test image: %d %%' % (100*correct/total))

