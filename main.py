# -*- coding: utf-8 -*-
"""
主函数 Main
"""

def train(**kwargs):
    '''
    训练
    '''
    
    pass


def val(model,dataloader):
    '''
    计算模型在验证集上的准确率
    '''
    pass


def test(**kwargs):
    '''
    测试
    '''
    pass


def help():
    '''
    打印帮助信息
    '''
    pass



'''
根据fire的使用方法
可通过python main.py <function> --args=xx的方式来执行训练或者测试。
'''

if __name__ == '__main__':
    import fire
    fire.Fire()