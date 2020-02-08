import pandas as pd
import numpy as np
import configparser
from trainlocal import *
import random
import os
import matplotlib
import matplotlib.pyplot as plt

def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    select = cf.items("select")
    epoch = dict(cf.items('select'))['epoch']
    dic = {}
    for key, val in select:
        dic[key] = val
    kup=int(dic['k_up'])
    kdown=int(dic['k_down'])
    model = [dic['model'] + cf.items("model")[0][1], dic['model'] + cf.items("model")[1][1]]
    addpath=dic['path']
    return kup,kdown,model,addpath
def get_result():
    num=20000
    rand = []
    for i in range(num):
        a = []
        start = random.uniform(0, 1)
        for _ in range(31):
            start += random.uniform(-0.1, 0.1)
            a.append(start)
        rand.append((np.array(a) - np.min(a)) / (np.max(a) - np.min(a)))


    return rand
def select_k():
    kup, kdown, model, path=parse_args('cfg.txt')
    for k in range(kdown,kup):
        axx, labels = [], []
        lab_dic = {'t': '正样本', 'f': '负样本'}
        for i in model:
            print('启用模型' + i)
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
        rand=get_result()
        lst = []
        print(k)
        for j, i in enumerate(rand):
            outputLabel = train().kNNClassify(i, dataSet, labels, k)
            if outputLabel[0][0] == '正':
                lst.append(i)
            if len(lst)==50:
                break
        print('识别出正样本' + str(len(lst)))
        # 识别出的正样本作图
        for j, i in enumerate(lst[:50]):
            t = np.arange(0, 31)
            s = i
            # Data for plotting
            fig, ax = plt.subplots()
            ax.plot(t, s)

            ax.set(xlabel='time (s)', ylabel='voltage (mV)',
                   title='About as simple as it gets, folks')
            ax.grid()
            if not os.path.exists(path +str(k)+'/'):
                os.makedirs(path +str(k)+'/')
            fig.savefig(path +str(k)+'/'+ str(j) + '.png')
            plt.cla()
            plt.close("all")
print(select_k())

