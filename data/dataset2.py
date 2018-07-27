# -*- coding: utf-8 -*-
"""
数据的相关处理
继承torch.utils.data.Dataset 的自定义数据集
"""
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import numpy as np
class DogCat(data.Dataset):
    def __init__(self,root,transforms = None,train = True,test = False):
        '''
            1 获取所有图片地址
            2 区分训练集和测试集
            3 从训练集中划出验证集
            3 予以不同的transform 
        '''
        self.test = test
        self.imgs = []
        
        # test: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        
        if test:
            self.imgs = [os.path.join(root,img) for img in os.listdir(root)]
        elif train:
            #训练集
            root_cat = root + 'cat/'
            self.imgs = [os.path.join(root_cat,img) for img in os.listdir(root_cat)]
            root_dog = root+ 'dog/'
            self.imgs += [os.path.join(root_dog,img) for img in os.listdir(root_dog)]
        else:
            #验证集
            root = 'data/valid/'
            root_cat = root + 'cat/'
            self.imgs = [os.path.join(root_cat,img) for img in os.listdir(root_cat)]
            root_dog = root + 'dog/'
            self.imgs += [os.path.join(root_dog,img) for img in os.listdir(root_dog)]
        
        if transforms == None:
            normalize = T.Normalize(mean=[0.485,0.456,0.406],
                                   std=[0.229,0.224,0.225])
            #测试集和验证集的数据转换
            if self.test or not train:
                self.transforms = T.Compose([
                                       T.Resize(224),
                                       T.CenterCrop(224),
                                       T.ToTensor(),
                                       normalize])

            else:
                self.transforms = T.Compose([
                                             T.Resize(224),
                                             T.CenterCrop(224),
                                             T.RandomHorizontalFlip(),
                                             T.ToTensor(),
                                             normalize])								
    
    
    def __getitem__(self,index):
        #需要返回单个图像的 data,label
        
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        
        #测试数据的label返回编号
        if self.test:
            label = int(img_path.split('.')[-2].split('/')[-1])
        else:
            # 1是狗 0是猫
            label = 1 if 'dog' in img_path else 0
        
        return data,label
        
    
    
    def __len__(self):
        return len(self.imgs)
                

'''
使用时 通过dataloader 加载数据
e.g.

train_dataset = data.dataset.DogCat(opt.train_data_root,train = True)
train_dataloader = torch.utils.data.Dataloader(train_dataset,
                                               batch_size = opt.batch_size,
                                               shuffle = True,
                                               num_workers=opt.num_workers)

for i,data in enumerate(trainloader):
    pass
    
    
'''
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

