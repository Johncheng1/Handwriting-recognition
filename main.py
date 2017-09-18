####create on September 18, 2017####
####author:wang cheng####
####mail:jsycwangc@163.com#####
import math
import numpy as np
from input_data import input_data
###测试加载mnist数据集是否成功###
print("read mnist database successfully")

######读取mnist数据集的数据####
sample, label, test_s, test_l = input_data()
sample = np.array(sample,dtype='float')
sample/= 256.0       # 特征向量归一化
test_s = np.array(test_s,dtype='float')
test_s /= 256.0

#####初始化神经网络参数#####
sam_num=len(sample)#样本总数
ipt_num=len(sample[0])#输入层神经元数
opt_num=10#输出层神经元数
hid_num=28#隐含层神经元数
w1=0.2*np.random.random((ipt_num,hid_num))-0.1#输入层和隐含层之间的权值初始化
w2=0.2*np.random.random((hid_num,opt_num))-0.1#输入层和隐含层之间的权值初始化
hid_offset_num=np.zeros(hid_num)#隐含层偏置
opt_offset_num=np.zeros(opt_num)#输出层偏置
ipt_lrate=0.2#输入层的学习率
hid_lrate=0.2#隐含层的学习率

##########sigmod激活函数定义#####
def active_function(x):
     function_vec=[]
     for i in x:
        function_vec.append(1/(1+math.exp(-i)))
     function_vec=np.array(function_vec)
     return function_vec


#########训练神经网络模型######
for count in range(0, sam_num):
    t_label = np.zeros(opt_num)
    t_label[label[count]] = 1
    #####正向传播#####
    hid_value=np.dot(sample[count],w1)+hid_offset_num#隐含层的n
    hid_function=active_function(hid_value)#激活函数  隐含层的输出
    opt_value=np.dot(hid_function,w2)+opt_offset_num#输出层的n
    opt_function = active_function(opt_value)  # 激活函数  输出层的输出
    #####反向传播#####
    e=t_label-opt_function#opt_value期望值和实际输出值的误差
    opt_delta=e*opt_function*(1-opt_function) #输出层和隐含层之间的敏感值
    correct_w2=0
    for i in range(0,opt_num):
        w2[:,i]+=hid_lrate*opt_delta[i]*hid_function#修改隐含层和输出层之间的权值
        correct_w2 +=1
    hid_delta=hid_function*(1-hid_function)*np.dot(w2,opt_delta)#隐含层和输入层之间的敏感值
    correct_w1 = 0
    for i in range(0,hid_num):
        w1[:,i]+=ipt_lrate*hid_delta[i]*sample[count]#修改隐含层和输入层之间的权值
        correct_w1 +=1
    opt_offset_num+=hid_lrate*opt_delta
    hid_offset_num+=ipt_lrate*hid_delta
print("train successfully")
print("Please hold on")
##############################################################
#####训练精度##########
success=0
for count in range(sam_num):
    hid_value = np.dot(sample[count], w1) + hid_offset_num  # 隐含层的n
    hid_function = active_function(hid_value)  # 激活函数  隐含层的输出
    opt_value = np.dot(hid_function, w2) + opt_offset_num  # 输出层的n
    opt_function = active_function(opt_value)  # 激活函数  输出层的输出
    if np.argmax(opt_function)==label[count]:
        success+=1
print('Training Accuracy is: %.2f%%'%((float(success)/len(sample))*100))
#####################################################################

######测试精度#########
true=0
for count in range(len(test_s)):
    hid_value = np.dot(test_s[count], w1) + hid_offset_num  # 隐含层的n
    hid_function = active_function(hid_value)  # 激活函数  隐含层的输出
    opt_value = np.dot(hid_function, w2) + opt_offset_num  # 输出层的n
    opt_function = active_function(opt_value)  # 激活函数  输出层的输出
    if np.argmax(opt_function) == test_l[count]:
        true+=1
print("Tset successfully")
print('Test Accuracy is: %.2f%%'%((float(true)/len(test_s))*100))