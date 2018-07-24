# -*- coding: utf-8 -*-
"""
数据的相关处理
继承torch.utils.data.Dataset 的自定义数据集
"""
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

class DogCat(data.Dataset):
    def __init__(self,root,transforms = None,train = True,test = False):
        '''
            1 获取所有图片地址
            2 区分训练集和测试集
            3 从训练集中划出验证集
            3 予以不同的transform 
        '''
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        self.test = test
        #整理文件次序
        # test: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs,key = lambda x :int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs,key = lambda x :int(x.split('.')[-2]))
        
        #获取图片数量
        imgnum = len(imgs)
        
        if test:
            self.imgs = imgs
        elif train:
            #前70%作为训练集
            self.imgs= imgs[:int(0.7*imgnum)]
        else:
            #后30%作为验证集
            self.imgs = imgs[int(0.7*imgnum):]
        
        if transforms == None:
            normalize = T.Normalize(mean=[0.485,0.456,0.406],
                                   std=[0.229,0.224,0.225])
            #测试集和验证集的数据转换
            if self.test or not train:
                self.transforms = T.Compose([normalize,
                                       T.Resize(224),
                                       T.CenterCrop(224),
                                       T.ToTensor()])
            else:
                self.transforms = T.Compose([normalize,
                                             T.Resize(256),
                                             T.CenterCrop(256),
                                             T.ToTensor()])
    
    
    def __getitem__(self,index):
        #需要返回单个图像的 data,label
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        
        #测试数据的label返回编号
        if self.test:
            label = img_path.split('.')[-2].split('/')[-1]
        else:
            # 1是狗 0是猫
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        
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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
