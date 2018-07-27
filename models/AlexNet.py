# -*- coding: utf-8 -*-
"""
AlexNet
"""

from torch import nn 
from .BasicModule import BasicModule

#图像为224*224

class AlexNet(BasicModule):
    def __init__(self,num_class = 2):
        super(AlexNet,self).__init__()
        
        self.model_name = 'alexnet'
        
        self.features = nn.Sequential(
                #第一次卷积
                nn.Conv2d(3,64,kernel_size = 11,stride = 4,padding = 2),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3,stride = 2),
                
                #第二次卷积
                nn.Conv2d(64,192,kernel_size = 5,padding = 2),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3,stride = 2),
                
                #第三次卷积
                nn.Conv2d(192,384,kernel_size = 3,padding = 1),
                nn.ReLU(inplace = True),
                
                #第四次卷积
                nn.Conv2d(384,256,kernel_size = 3,padding = 1),
                nn.ReLU(inplace = True),
                
                #第五次卷积
                nn.Conv2d(256,256,kernel_size = 3,padding = 1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3,stride =2),
                
                )
        self.classfier = nn.Sequential(
                #第一层全连接 
                nn.Dropout(),
                nn.Linear(256*6*6,4096),
                nn.ReLU(inplace = True),
                #第二层全连接
                nn.Dropout(),
                nn.Linear(4096,4096),
                nn.ReLU(inplace = True),
                #第三层全连接
                nn.Linear(4096,num_class),
                
                )
        
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),256*6*6)
        x = self.classfier(x)
        return x
        
        
        
        
        