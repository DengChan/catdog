# -*- coding: utf-8 -*-
"""
主函数 Main
"""
import os
import torch as t
from torch.utils.data import DataLoader
import torchvision as tv
import models
from torchnet import meter
from tqdm import tqdm

from data.dataset import DogCat
from config import opt
from utils.visualize import Visualizer



'''
    训练函数的步骤
    1. 定义网络模型
    2. 加载训练和验证数据
    3. 定义损失和优化函数
    4. 定义平均损失和混淆矩阵
    5. 训练
        5.1 训练模型
        5.2 计算损失，更新混淆矩阵
        5.3 可视化损失
    6. 模型应用于验证集，可视化结果
    
 '''

def train(**kwargs):
    
    '''根据cmd输入更新配置 '''
    opt.parse(kwargs)
    '''初始化可视化工具'''
    vis = Visualizer(opt.env)
    
    '''定义网络模型'''
    model = getattr(models,opt.model)()
    #加载指定的模型参数
    if opt.load_model_path :
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.cuda()
    
    '''加载数据'''
    train_dataset = DogCat(opt.train_data_root,train = True)
    train_loader = DataLoader(train_dataset,batch_size = opt.batch_size,
                              shuffle = True,num_workers = opt.num_workers)
    # test 默认为false 两者都为false则为验证集
    valid_dataset = DogCat(opt.train_data_root,train = False)
    valid_loader = DataLoader(valid_dataset,batch_size = opt.batch_size,
                              shuffle = False,num_workers = opt.num_workers)
    
    '''定义损失和优化函数'''
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,
                             weight_decay = opt.weight_decay)
    
    '''定义平均损失和混淆矩阵'''
    #meter自动平均损失，每次只需要通过add方法添加新数据
    loss_meter = meter.AverageValueMeter()
    
    '''
    混淆矩阵格式
    
        被判断为狗   被判定为猫
    狗    
    
    猫
    
    '''
    confusion_matrix = meter.ConfusionMeter(2)
    #保存经过上一轮epoch训练后的loss ， 比较两次loss,进而判断是否继续训练模型
    previous_loss = 1e100
    
    '''训练'''
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        for i,data in enumerate(train_loader):
            inputs,labels = data
            #是否使用GPU计算
            if opt.use_gpu :
                inputs = inputs.cuda()
                labels = labels.cuda()
            
            #训练网络
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            #更新各指标以及可视化
            loss_meter.add(loss.data[0])
            print('loss.data[0] = ',loss.data[0])
            print('loss.item() = ',loss.item())
            confusion_matrix.add(outputs.data,labels.data)
            
            if i%opt.print_freq == opt.print_freq-1:
                print('loss_meter.value()[0] = ',loss_meter.value()[0])
                vis.plot('loss',loss_meter.value()[0])
                
                #进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()
        #保存模型    
        model.save()
        
        #在验证集上验证，并将结果可视化
        #验证集的混淆矩阵，预测精度
        valid_cm,valid_accuracy = valid(model,valid_loader)
        
        vis.plot('valid_accuracy',valid_accuracy)
        vis.log("epoch:{epoch},lr:{lr},train_cm:{train_cm},valid_cm:{valid_cm},".format(
                epoch = epoch,lr = lr,train_cm = str(confusion_matrix.value()),valid_cm = str(valid_cm.value())))
        
        #更新lr
        if loss_meter.value()[0]>previous_loss:
            lr = lr * opt.lr_decay
            for lr,param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        previous_loss = loss_meter.value()[0]
        
        
    
    
    


def valid(model,dataloader):
    '''
    计算模型在验证集上的准确率
    '''
    '''计算预测数据'''
    #验证模式，使模型参数保持不变
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    
    for i,data in enumerate(dataloader):
        inputs,labels = data
        if opt.use_gpu :
            inputs = inputs.cuda()
        outputs = model(inputs)
        '''修改点'''
        confusion_matrix.add(outputs.data,labels.data)
    
    #恢复为训练模型
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * ((cm_value[0][0]+cm_value[1][1])/cm_value.sum())
    return confusion_matrix,accuracy    


def test(**kwargs):
    '''
    测试
    '''
    #加载配置
    opt.parse(kwargs)
    model = getattr(models,opt.model)()
    if opt.load_model_path :
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.cuda()
    
    #加载数据
    test_dataset = DogCat(opt.test_data_root,test = True)
    test_loader = DataLoader(test_dataset,batch_size=opt.batch_size,
                             shuffle = False,num_workers = opt.num_workers)
    results = []
    for i,data in enumerate(test_loader):
        inputs,labels = data
        if opt.use_gpu:
            inputs = inputs.cuda()
        outputs = model(inputs)
        #softmax将计算结果归一化为概率，batch_results记录每一组数据的（编号，概率）字典
        probability = t.nn.functional.softmax(outputs)[:,0].tolist()
        batch_results = [(id_num,proba) for id_num,proba in zip(labels,probability)]
        results += batch_results
    
    write_csv(results,opt.result_file)

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerow(results)


def help():
    '''
    打印帮助信息 通过命令台命令 python file.py help
    '''
    
    print("""
          usage:python file.py <function> [--args=value]
          <function>:= train | test | help 
          example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
          avaiable args:""".format(__file__))
        
    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


'''
根据fire的使用方法
可通过python main.py <function> --args=xx的方式来执行训练或者测试。
'''

if __name__ == '__main__':
    import fire
    fire.Fire()