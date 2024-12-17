from matplotlib import pyplot as plt
import numpy as np

import torch
# from torch import nn

# 判断迭代需不需要停止的类
class EarlyStopping_loss():
    def __init__(self, patience = 5, tol = 0.0005): # 惯例地定义我们所需要的一切变量/属性
        # 当连续patience次迭代时，这一轮迭代的损失与历史最低损失之间的差值小于阈值时，触发提前停止
        
        self.patience = patience
        self.tol = tol # tolerance，累积5次都低于tol才会触发停止
        self.counter = 0 # 计数，计算现在已经累积了counter次
        self.lowest_loss = None
        self.early_stop = False # True - 提前停止，False - 不要提前停止
    
    def __call__(self,val_loss):
        if self.lowest_loss == None: # 这是第一轮迭代
            self.lowest_loss = val_loss
        elif (self.lowest_loss - val_loss) > self.tol:
            self.lowest_loss = val_loss
            self.counter = 0
        elif (self.lowest_loss - val_loss) < self.tol:
            self.counter += 1
            print("\t NOTICE: Early stopping counter {} of {}".format(self.counter,self.patience))
            if self.counter >= self.patience:
                print('\t NOTICE: Early Stopping Actived')
                self.early_stop = True
        return self.early_stop
        # 这一轮迭代的损失与历史最低损失之间的差 - 阈值

# 判断迭代需不需要停止的类
class EarlyStopping_accuracy():
    def __init__(self, patience = 5, tol = 0.0005): # 惯例地定义我们所需要的一切变量/属性
        # 当连续patience次迭代时，这一轮迭代的损失与历史最低损失之间的差值小于阈值时，触发提前停止
        
        self.patience = patience
        self.tol = tol # tolerance，累积5次都低于tol才会触发停止
        self.counter = 0 # 计数，计算现在已经累积了counter次
        self.highest_accuracy = None
        self.early_stop = False # True - 提前停止，False - 不要提前停止
    
    def __call__(self,val_accuracy):
        if self.highest_accuracy == None: # 这是第一轮迭代
            self.highest_accuracy = val_accuracy
        elif (val_accuracy - self.highest_accuracy) > self.tol:
            self.highest_accuracy = val_accuracy
            self.counter = 0
        elif (val_accuracy - self.highest_accuracy) < self.tol:
            self.counter += 1
            print("\t NOTICE: Early stopping counter {} of {}".format(self.counter,self.patience))
            if self.counter >= self.patience:
                print('\t NOTICE: Early Stopping Actived')
                self.early_stop = True
        return self.early_stop
        # 这一轮迭代的损失与历史最低损失之间的差 - 阈值


def IterOnce(model,criterion,optimizer,x,targets):
    """
    对模型进行一次迭代的函数
    
    net: 实例化后的架构
    criterion: 损失函数
    opt: 优化算法
    x: 这一个batch中所有的样本
    targets: 这一个batch中所有样本的真实标签
    """
    [preds] = model(x)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True) #比起设置梯度为0，让梯度为None会更节约内存

    yhat = torch.max(preds,1)[1]
    correct = torch.sum(yhat == targets)

    return correct, loss


def TestOnce(model,criterion,x,targets):
    """
    对一组数据进行测试并输出测试结果的函数
    
    net: 经过训练后的架构
    criterion: 损失函数
    x: 要测试的数据的所有样本
    y: 要测试的数据的真实标签
    """
    #对测试，一定要阻止计算图追踪
    #这样可以节省很多内存，加速运算
    with torch.no_grad(): 
        [preds] = model(x)
        loss = criterion(preds,targets)
        yhat = torch.max(preds,1)[1]
        correct = torch.sum(yhat == targets)
    return correct,loss


#绘图函数
def plot_loss_accuracy(trainloss, testloss, trainaccuracy, testlaccuracy):
    epochs = np.arange(0,len(trainloss),1)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(epochs, trainloss, color='red', label='Train loss')
    plt.plot(epochs, testloss, color='orange', label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    epochs = np.arange(0,len(trainaccuracy),1)
    plt.plot(epochs, trainaccuracy, color='blue', label='Train accuracy')
    plt.plot(testlaccuracy, color='green', label='Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


