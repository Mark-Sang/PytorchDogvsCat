import warnings
import torch

class DefaultConfig():
    env = "default"                                             #visdom环境
    vis_port = 8097                                             #visdom端口
    model = "ResNet34"                                          #使用的模型
    train_data_root = "./train"                                 #训练集存放路径
    test_data_root = "./test"                                   #测试集存放路径
    load_model_path = None                                      #加载预训练模型的路径，None表示不加载
    batch_size = 8                                              #batch_size
    use_gpu = True
    num_workers = 4
    print_freq = 50
    debug_file = ""
    result_file = "./submission.txt"
    max_epoch = 6
    lr = 0.001
    lr_decay = 0.95
    weight_decay = 1e-4                                         #损失函数

    def parse(self, kwargs):                                    #根据字典更新config参数，便于命令行更改参数
        for k,v in kwargs.items():                              #更新配置参数
            if not hasattr(self, k):
                warnings.warn("Warning:设置文件里没这个参数")
            setattr(self, k, v) 
            
        print("user config:")
        for k,v in self.__class__.__dict__.items():
            if (not k.startswith('__') and not k.startswith('p')):                
                print(k,getattr(self, k))
        print('\n''\n')