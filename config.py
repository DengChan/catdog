# -*- coding: utf-8 -*-
"""

保存一些变量的默认值
一般为opt实体

"""

import warnings

class DefaultConfig(object):
    env = 'default' #visdom环境的默认值
    model = 'AlexNet' #使用的模型名称，必须是在models/__init__.py中声明过的模型
    
    train_data_root = 'train/'
    test_data_root = 'test/'
    #load_model_path = 'checkpoints/model.pth'#预训练模型
    load_model_path = ''
    debug_file = '/tmp/debug'
    result_file = 'result.csv'
    
    batch_size = 128
    use_gpu = True
    num_workers = 4
    print_freq = 20 # 每多少组batch就输出一次信息
    max_epoch = 10
    lr = 0.1
    lr_decay = 0.95 # lr = lr * lr_decay
    weight_decay = 1e-4
    
    '''根据字典更新配置属性操作'''
    def parse(self,kwargs):
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning:opt has npt attribute %s" %k)
            
            setattr(self,k,v)
        
        #打印配置信息
        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if not k[0:2] == '__':
                print(k,getattr(self,k))


#DefaultConfig.parse = parse
opt = DefaultConfig()