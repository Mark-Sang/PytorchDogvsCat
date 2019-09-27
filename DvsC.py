from dataload import DogCat
from config import DefaultConfig
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from torchvision import models
import torch 
import torch.nn as nn
import numpy as np
import os
import csv
import fire
import visdom 

opt = DefaultConfig()                                  #参数配置opt
vis_loss = []                                          #vis_loss数组

def help():
    print('There are help')
    print('It is a test')

def train(**kwargs):                                   #训练模型：定义网络，定义数据，定义损失函数和优化器，训练并计算指标，计算验证集上的准确率    
    opt.parse(kwargs)                                  #根据命令行参数更新配置
    vis_num = 0                                        #vis计数
    vis = visdom.Visdom(env=opt.env)

    """(1)加载网络，若有预训练模型也加载"""
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512,2)
    if opt.use_gpu:
        model.cuda()

    """"(2)处理数据"""
    train_data = DogCat(opt.train_data_root, train=True)    #训练集
    val_data = DogCat(opt.train_data_root, train=False)     #验证集

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    """"(3)定义损失函数和优化器"""
    criterion = torch.nn.CrossEntropyLoss()                 #交叉熵损失
    lr = opt.lr                                             #学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    """(4)统计指标，平滑处理之后的损失，还有混淆矩阵"""
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    """(5)-1开始训练"""
    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in enumerate(train_dataloader):
            
            input = Variable(data)                         #训练模型参数
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()
            

            optimizer.zero_grad()                          #梯度清零
            score = model(input)

            loss = criterion(score, target)
            loss.backward()                                #反向传播
            
            
            optimizer.step()                               #更新参数
            loss_meter.add(loss.item())                    #更新统计指标及可视化
            confusion_matrix.add(score.detach(), target.detach())
            print("ii:",ii,"loss:",loss_meter.value()[0])
            if ii%opt.print_freq == opt.print_freq-1:    
                vis_loss.append(loss_meter.value()[0])
                vis_num = vis_num+1

        vis.bar(vis_loss)                                  #visdom显示
       # vis.text(vis_num)

        """(5)-2模型参数存储"""
        torch.save(model.state_dict(),"./checkpoint/"+"model.pth")    
    
        """(5)-3计算验证集上的指标及可视化"""
        val_cm,val_accuracy = val(model,val_dataloader)
        vis.text(val_accuracy)

        """(5)-4如果损失不再下降，则降低学习率"""
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):                               #计算模型在验证集上的准确率等信息
    model.eval()                                       #将模型设置为验证模式

    confusion_matrix = meter.ConfusionMeter(2)
    for ii,data in enumerate(dataloader):
        input,label = data
        val_input = Variable(input, volatile=True)
        val_lable = Variable(label.long(), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_lable = val_lable.cuda()

        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.long())

    model.train()                                      #模型恢复为训练模式
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]/cm_value.sum())

    return confusion_matrix,accuracy
        

def test(**kwargs):
    opt.parse(kwargs)                                  #根据命令行参数配置更新
  
    test_data = DogCat(opt.test_data_root,test=True)   #data
    test_dataloder = DataLoader(test_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []

    model = models.resnet34(pretrained=True)           #model
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load('./checkpoint/model.pth'))
    
    if opt.use_gpu:
        model.cuda()
    model.eval()

    for ii,(data, path) in enumerate(test_dataloder):
        input = Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        path = path.numpy().tolist()
        _ds,predicted = torch.max(score.data, 1)
        predicted = predicted.data.cpu().numpy().tolist()
        res = ""
        for (i,j) in zip(path, predicted):
            if j==1:
                res="Dog"
            else:
                res="Cat"
            results.append([i,"".join(res)])
    
    write_csv(results,opt.result_file)
    return results

def write_csv(results, file_name):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

if __name__ == '__main__':    
    fire.Fire() 



    
