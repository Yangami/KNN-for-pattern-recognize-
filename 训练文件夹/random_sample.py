import pandas as pd
import winsound
import numpy as np
import configparser
from trainlocal import *
import random
import os
import matplotlib
import matplotlib.pyplot as plt
#读配置文件获取随即数量、k值、路径
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    rand = cf.items("rand")
    epoch = dict(cf.items('init'))['epoch']
    dic = {}
    for key, val in rand:
        dic[key] = val
    k=int(dic['k'])
    num=int(dic['num'])
    picfile = dic['newpicfile']
    model = [dic['model'] + cf.items("model")[0][1], dic['model'] + cf.items("model")[1][1]]
    sampfile = dic['newsampfile']
    # 从 init 中获取path
    if dic['follow']:
        assert 'rand/new_pic1' in dic['newpicfile']
        assert 'model/model'in dic['model']
        picfile = picfile[:picfile.rindex('new_pic') + 7] + str(epoch) + '/'
        model=[model[0][:model[0].rindex('/model')+6]+str(epoch)+ '/'+cf.items("model")[0][1],
            model[0][:model[0].rindex('/model')+6]+str(epoch)+'/'+ cf.items("model")[1][1]]
        sampfile=sampfile[:sampfile.rindex('/')]+'/'+str(epoch)+'.txt'
    return k,num,sampfile,picfile,model,epoch
k,num,sampfile,picfile,model,epoch=parse_args('cfg.txt')
path=model+[sampfile]
path=path+[picfile]
print(path)
for p in path:
    try:
        p = p[:p.rindex('/')]
    except:
        continue
    if not os.path.exists(p):
        os.makedirs(p)
        print(p + ' 创建成功')
    else:
        print(p + ' 目录已存在')
#生成随机序列
rand=[]
for i in range(num):
    a=[]
    start = random.uniform(0, 1)
    for _ in range(31):
        start+=random.uniform(-0.13,0.13)
        a.append(start)
    rand.append((np.array(a)-np.min(a))/(np.max(a)-np.min(a)))
#获取模型
#dataSet,labels=train().get_dataSet()
axx, labels = [], []
dir=model
lab_dic = {'t': '正样本', 'f': '负样本'}
for i in dir:
    print('启用模型'+i)
    if 'b' in [str(open(i).read())[0], str(open(i).read())[1]]:  # ,'/home/jquser/'+
        lines = str(open(i).read())[2:-1].split('\\n')
    else:
        lines = open(i).read().split('\n')
    f = []
    for line in lines:
        if len(line) != 0:
            axx.append([float(j) for j in line.split(',')])
            labels.append(lab_dic[i[i.find('parameter') - 2]])
dataSet, labels = np.array(axx), labels  # createDataSet()
#识别出正样本

lst = []
fnum=0
print(k)
for j,i in enumerate(rand):
    if j%(len(rand)//10)==0:
        print('正在识别第'+str(j)+'个随机序列')
    outputLabel = train().kNNClassify(i, dataSet, labels, k)
    if outputLabel[0][0]=='正':
        #if outputLabel[1]['负样本']==(k//2)+(k%2)-1:
        lst.append(i)
    # elif '正样本'in outputLabel[1]:
    #
    #     if outputLabel[1]['正样本']==(k//2)+(k%2)-1:
    #         lst.append(i)
    #         fnum += 1
    #     elif outputLabel[1]['正样本']==(k//2)+(k%2)-2:
    #         lst.append(i)
    #         fnum += 1
    #if '正样本'in outputLabel[1].keys():
        #print(outputLabel)
print('识别出正样本'+str(len(lst)-fnum)+','+str(fnum))
with open(sampfile,'a') as f:
    if len(lst)!=0:
        for i in lst:
            f.write(','.join([str(_)for _ in i])+'\n')
#识别出的正样本作图
for j,i in enumerate(lst):
    t = np.arange(0, 31)
    s=i
    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='About as simple as it gets, folks')
    ax.grid()
    fig.savefig(picfile + str(j) + '.png')
    # if not os.path.exists('./rand/newpic18-'+str(k)+'/'):
    #     os.makedirs('./rand/newpic18-'+str(k)+'/')
    #     print('./rand/newpic18-'+str(k)+'/' + ' 创建成功')
    # fig.savefig('./rand/newpic18-'+str(k)+'/'+str(j)+'.png')
    plt.cla()
    plt.close("all")
with open('./label/label'+str(epoch)+'.txt','a') as d:
    d.write('正样本'+'\t'+'\n')
    d.write('负样本'+'\t'+'1,2')
winsound.Beep(400,1000)

