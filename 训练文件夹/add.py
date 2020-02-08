import pandas as pd
import numpy as np
import configparser
from trainlocal import *
from sklearn.cluster import DBSCAN
import random
import os
import matplotlib
import matplotlib.pyplot as plt
#读配置文件获取随即数量、k值、路径
def parse_args(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    epoch = int(dict(cf.items('init'))['epoch'])
    add = cf.items("add")
    dic = {}
    for key, val in add:
        dic[key] = val
    labelpath = dic['labelpath']
    truepath = dic['true']
    falsepath = dic['false']
    dic1 = {}
    for key, val in cf.items('rand'):
        dic1[key] = val
    model = [dic1['model'] + cf.items("model")[0][1], dic1['model'] + cf.items("model")[1][1]]
    if dic['follow']:
        labelpath=labelpath[:labelpath.find('.txt')-1]+str(epoch)+'.txt'
        truepath=truepath[:truepath.find('.txt')-1]+str(epoch+1)+'.txt'
        falsepath = falsepath[:falsepath.find('.txt') - 1] + str(epoch+1) + '.txt'
        model = [model[0][:model[0].rindex('/model') + 6] + str(epoch) + '/' + cf.items("model")[0][1],
                 model[0][:model[0].rindex('/model') + 6] + str(epoch) + '/' + cf.items("model")[1][1]]
    faddnum=int(dic['faddnum'])
    taddnum = int(dic['taddnum'])
    k=int(dic1['k'])
    rand=cf.items("rand")
    for key, val in rand:
        dic[key] = val
    file = dic['newsampfile']
    if dic['follow']:
        file=file[:file.find('.txt') - 1] + str(epoch) + '.txt'
    return epoch,truepath,falsepath,labelpath,file,k,taddnum,faddnum,model
epoch,truepath,falsepath,label,file,k,taddnum,faddnum,model=parse_args('cfg.txt')

#knn
def kNNClassify(newInput, dataSet, labels, k=32):
    dataSet = np.array(dataSet)
    numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
    diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
    squaredDiff = diff ** 2  # squared for the subtract
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
    distance = squaredDist ** 0.5

    sortedDistIndices = np.argsort(distance)
    distsum=0
    classCount = {}  # define a dictionary (can be append element)
    for i in range(k):
        ## step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]
        if voteLabel=='正样本':
            distsum+=distance[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    distsum=distsum/classCount['正样本']
    if '正样本' in list(classCount.keys()):
        if classCount['正样本'] == np.max(list(classCount.values())):
            maxIndex = '正样本'
        else:
            maxIndex = '负样本'
    else:
        maxIndex = '负样本'

    return maxIndex, classCount,distsum
def one_cluster(X,eps,min_samples):
    pred=DBSCAN(eps, min_samples).fit_predict(X)
    print(set(pred))
    if 0 not in set(pred):
        print('聚类1')
        eps+=0.001
        pred=one_cluster(X,eps,min_samples)
    if -1 not in pred:
        print('聚类2')
        eps-=0.01
        pred= one_cluster(X, eps, min_samples)
    return pred
def get_dbscan(fnum,axx):
    X=np.array(axx)
    y_pred = one_cluster(X,eps=0.7, min_samples=5)
    print(set(y_pred))
    dic={}

    for i in set(y_pred):
        dic[i]=[]
    for j,i in enumerate(y_pred):
        dic[i].append(j)
    maxk=0
    clu=_
    for key in set(y_pred):
        if key!=-1&len(dic[key])>maxk:
            maxk=len(dic[key])
            clu=key
    data=[]
    for i in dic[clu]:
        data.append(X[i])
    dist=[]
    for newInput in data:
        dataSet = np.array(data)
        numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
        diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
        squaredDiff = diff ** 2  # squared for the subtract
        squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
        distance = squaredDist ** 0.5
        dist.append(sum(distance))
    return list(np.argsort(dist)[:fnum]),len(dic)-1,len(dic[clu])
# 获取训练集
classing=open(file).readlines()
lblst=open(label).read().split('\n')
for j,i in enumerate(lblst):
    if '正样本' in i:
        tidx=j
    if '负样本' in i:
        fidx=j

dic={'true':list(map(int, (lblst[tidx][lblst[tidx].find('正样本')+4:]).split(','))),
     'false':list(map(int, (lblst[fidx][lblst[fidx].find('负样本')+4:]).split(',')))}
_=[]
for i in range(len(classing)):
    if i not in dic['true']:
        _.append(i)
dic['false']=_
print(dic)
#新建add文件夹
for p in [truepath,falsepath]:
    try:
        p = p[:p.rindex('/')]
    except:
        continue
    if not os.path.exists(p):
        os.makedirs(p)
        print(p + ' 创建成功')
    else:
        print(p + ' 目录已存在')

#获取模型
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

print('读',str(file))
falsesample,truesample=[],[]
for i in dic['false']:
    falsesample.append(classing[i])
for i in dic['true']:
    truesample.append(classing[i])
tsample = [list(map(float, r.split(','))) for r in truesample]
fsample = [list(map(float, r.split(','))) for r in falsesample]
try:
    np.array(fsample).reshape(-1, 31)
except:
    print('新样本长度不匹配')
print('负样本数',np.shape(fsample)[0])
#识别出正样本
def false1():
    lst=[]
    for j,i in enumerate(fsample):
        outputLabel = kNNClassify(i, dataSet, labels, k)
        print(outputLabel[1]['负样本'], outputLabel)
        if outputLabel[1]['负样本']==(k//2)+(k%2)-1:
            lst.append(i)

    if len(lst)<faddnum:
        for j, i in enumerate(fsample):
            outputLabel = kNNClassify(i, dataSet, labels, k)
            print(outputLabel[1]['负样本'], outputLabel[1])
            if outputLabel[1]['负样本'] == (k//2)+(k%2)-2:
                lst.append(i)
    return lst
def false2():
    lst=[]
    dist=[]
    for j,i in enumerate(fsample):
        outputLabel = kNNClassify(i, dataSet, labels, k)
        dist.append(outputLabel[2])
    for i in range(faddnum):
        lst.append(fsample[np.argsort(dist)[i]])
    return lst
def false3():
    tlst=[]
    result=[]
    tsort=[]
    for j,i in enumerate(fsample):
        outputLabel =kNNClassify(i, dataSet, labels, k)
        tlst.append(i)
        tsort.append(outputLabel[1]['正样本'])
    for i in range(faddnum):
        print(np.argsort(tsort))
        try  :
            result.append(tlst[np.argsort(tsort)[i]])
        except :
            continue
    print(tsort)
    return result
def false4():
    random.shuffle(fsample)
    return fsample[:faddnum]
def false_dbs():
    lst=[]
    idx,cluster,clusnum=get_dbscan(faddnum,fsample)
    for i in idx:
        lst.append(fsample[i])

    print('聚类结果：',idx,cluster,clusnum)
    return lst,idx,cluster,clusnum
lst,idx,cluster,clusnum=false_dbs()
#筛选正样本
tlst=[]
def mod_selet():
    for j,i in enumerate(tsample):
        outputLabel =kNNClassify(i, dataSet, labels, k)
        print(outputLabel[1]['负样本'], outputLabel)
        if outputLabel[1]['负样本']>=(k//2)+(k%2)-1:
            tlst.append(i)

    if len(tlst)<taddnum:
        for j, i in enumerate(tsample):
            outputLabel = kNNClassify(i, dataSet, labels, k)
            print(outputLabel[1]['负样本'], outputLabel[1])
            if outputLabel[1]['负样本'] == (k//2)-2+(k%2):
                tlst.append(i)
    # if len(tlst)<taddnum:
    #     for j, i in enumerate(tsample):
    #         outputLabel = train().kNNClassify(i, dataSet, labels, k)
    #         print(outputLabel[1]['负样本'], outputLabel[1])
    #         if outputLabel[1]['负样本'] == (k // 2) - 3:
    #             tlst.append(i)
def singl_selet():
    result=[]
    tsort=[]
    for j,i in enumerate(tsample):
        outputLabel =kNNClassify(i, dataSet, labels, k)

        tlst.append(i)
        print('t',outputLabel)
        if  '负样本'in outputLabel[1].keys():
            tsort.append(outputLabel[1]['负样本'])
        else:
            tsort.append(0)
    print(tsort)
    for i in range(max(len(tlst),taddnum)):
        try  :
            result.append(tlst[np.argsort(tsort)[i]])
        except :
            print('wrong',i)
            continue
    return result
def mid_select():
    result=[]
    for i,j in enumerate(tsample):
        outputLabel = kNNClassify(j, dataSet, labels, k)
        tnum=outputLabel[1]['正样本']
        if tnum in range((k//3),(2*k//3)+1):
            result.append(j)
    return result
def good_selet():
    tlst=[]
    for j,i in enumerate(tsample[:taddnum]):
        tlst.append(i)
    return tlst
def false_selet():
    tlst=[]
    for i in tsample:
        outputLabel = kNNClassify(i, dataSet, labels, k)
        if outputLabel[0][0]=='负':
            tlst.append(i)
    return tlst
tlst=mid_select()

test_acc=len(tsample)/(len(tsample)+len(falsesample))
print(len(tsample),len(tlst))
if __name__ == "__main__":
    print('符合要求负样本数量：',len(fsample),len(lst))
    with open(falsepath,'a') as f:
        print('epoch=',epoch)
        print('写入：'+str(falsepath),'训练集已更新 请开始新一轮训练')
        if len(lst)!=0:
            for i in lst[:]:
                f.write(','.join([str(_)for _ in i])+'\n')
    f.close()
    with open(truepath,'a') as f:
        for i in tlst[:taddnum]:
            f.write(','.join([str(_)for _ in i])+'\n')
    f.close()
    with open('train_log.txt','a') as l:
        print(str(epoch)+',testacc='+str(test_acc))
        l.write('epoch'+str(epoch)+',testacc='+str(test_acc)+' '+str(len(tsample)+len(fsample))+',k='+str(k)+'cluster'+str(idx)+','+str(cluster)+','+str(clusnum)+'falsedbs_signal'+'\n')
    l.close()