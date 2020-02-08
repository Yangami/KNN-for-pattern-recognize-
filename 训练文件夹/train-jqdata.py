import pandas as pd
import numpy as np
import configparser
import datetime
import time
def get_id():
    cf = configparser.ConfigParser()
    cf.read('cfg.txt')
    jqcount = cf.items("jqcount")
    count = []
    for k, v in jqcount:
        count.append(v)
    return count[0],count[1]
id,k=get_id()
# 若本地训练 要安装jqdatasdk包
from jqdatasdk import *
auth(id,k)


beg=time.time()
# KNN部分函数定义
class train():
    def _init_(self):
        pass

    def parse_args(self,filename):
        cf = configparser.ConfigParser()
        cf.read(filename)
        # 查看所有节点
        # secs = cf.sections()
        # init section
        init = cf.items("init")
        dic = {}
        for k, v in init:
            dic[k] = v
        # 从 init 中获取path
        modelpath =[dic[dic['model_path']]+cf.items('model')[0][1],dic[dic['model_path']]+cf.items('model')[1][1]]
        trainpath=dic['train_path']
        rdatapath=dic['rdata_path']
        files = cf.items('train')#训练文件
        t_file,f_file=files[0][1],files[1][1]
        val=cf.items('valid')[0][1]#验证文件
        rdata = [rdatapath+i[1] for i in cf.items('rawdata')]

        # read
        dirs = [trainpath + t_file, trainpath + f_file,trainpath+val,rdata,modelpath]
        return dirs
    def createDataSet(self):
        # 建立构建分类器的原始样本
        data_dic={'x'}
        group = array(data_dic['x'])
        labels = data_dic['y']  # four samples and two classes
        return group, labels

    def kNNClassify(self, newInput, dataSet, labels, k=32):
        dataSet = np.array(dataSet)
        numSamples = dataSet.shape[0]  # shape[0] stands for the num of row
        diff = np.tile(newInput, (numSamples, 1)) - dataSet  # Subtract element-wise
        squaredDiff = diff ** 2  # squared for the subtract
        squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row
        distance = squaredDist ** 0.5

        sortedDistIndices = np.argsort(distance)

        classCount = {}  # define a dictionary (can be append element)
        for i in range(k):
            ## step 3: choose the min k distance
            voteLabel = labels[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        ## step 5: the max voted class will return
        maxCount = 0
        for key, value in classCount.items():
            if value > maxCount:
                maxCount = value
                maxIndex = key
        return maxIndex, classCount
    def get_data(self,security, start_date, end_date, count):
        try:
            df = get_price(str(security), start_date=start_date, end_date=end_date)
        except:
            return[]
        close= df['close']
        return close

    def get_dataSet(self, compress_size=30):
        axx, labels = [], []
        (truedir, falsedir, dirs, rfilelst, _) = self.parse_args('cfg.txt')
        rdf = pd.read_csv(rfilelst[0], index_col=0)
        for i in rfilelst[1:]:
            df1 = pd.read_csv(i, index_col=0)
            rdf = pd.concat([rdf, df1], axis=1)
        dir = [truedir, falsedir]
        lab_dic = {truedir: '正样本', falsedir: '负样本'}
        for name in dir:
            # print('read', name)
            f = open(name, encoding='utf-8').readlines()

            for j, i in enumerate(f[:-1]):
                if i[0] == '6':
                    security = i[0:6] + '.XSHG'
                else:
                    security = i[0:6] + '.XSHE'
                start_date = i[7:17]
                end_date = i[18:28]
                x = []
                close = self.get_data(security=security, start_date=start_date, end_date=end_date,count=None)
                close=list(close)
                if len(close) != 0:
                    df = pd.DataFrame({'close': close}, index=[i for i in range(len(close))])
                    df = df.dropna()
                len(df) == 0
                try :
                    len(df) == 0
                except:
                    print(security,start_date,end_date)
                    continue
                df['close'] = (df['close'] - np.min(df['close'])) / (np.max(df['close']) - np.min(df['close']))
                x.append(df['close'][0])
                for i in range(compress_size):
                    x.append(df['close'][max(int((i + 1) * (len(df) / compress_size)) - 1, 0)])
                axx.append(list(x))
                labels.append(lab_dic[name])
        dataSet, labels = np.array(axx), labels  # createDataSet()
        return dataSet, labels

    # 获取W型股票列表,dirs格式security startdate enddate
    def get_wst(self,compress_size,k):
        pred = []
        (truedir, falsedir, dirs, rfilelst,_) = self.parse_args('cfg.txt')

        dataSet,labels=self.get_dataSet()
        f = open(dirs,encoding='utf-8').readlines()
        for j, i in enumerate(f[:]):
            test_x=[]
            if i[0]=='6':
                security = i[:6] + '.XSHG'
            else:
                security = i[:6] + '.XSHE'
            start_date = i[7:17]
            end_date = i[18:28]
            x = []
            close = self.get_data(security=security, start_date=start_date,end_date=end_date ,count=None)
            df = pd.DataFrame({'close': close})
            df = df.dropna()
            if len(df) == 0:
                continue
            df['close'] = (df['close'] - np.min(df['close'])) / (np.max(df['close']) - np.min(df['close']))
            x.append(df['close'][0])
            for i in range(compress_size):
                x.append(df['close'][max(int((i + 1) * (len(df) / compress_size)) - 1, 0)])
            test_x.append(list(x))
            n = 0
            for i in test_x:
                for j in i:
                    if np.isnan(j):
                        n = 1
            if n == 1:
                    continue
            for j, i in enumerate(test_x):
                testX = i
                outputLabel, tnum = self.kNNClassify(testX, dataSet, labels, k)
                if outputLabel[0] == '正':
                    # outputLabel = KNN_inM(testX, a, labs,20 )
                    pred.append(1)
                    print( " classified to class: ",outputLabel)
                else:
                    pred.append(0)
                    print( " classified to class: ",outputLabel)
        corr=0
        for i,j in enumerate(pred):
            if (i<120)&(j==1):
                corr+=1
            elif (i>=150)&(j==0):
                corr+=1
        acc=corr/len(pred)
        return acc
    def get_model(self,dataSet,labels):
        (_,_,_,_,path) = self.parse_args('cfg.txt')
        for i,j in enumerate(labels):
            if j=='负样本':
                with open(str(path[0]),'a') as u:

                    u.write(','.join([str(k) for k in dataSet[i]])+'\n')
            else:
                with open(str(path[1]),'a') as u:
                    u.write(','.join([str(k) for k in dataSet[i]])+'\n')
    def op(self):
        open(truedir)
#训练并计算验证集准确率
pred=train().get_wst(30,28)
print(pred)
#保存模型
a=train()
a.get_model(a.get_dataSet()[0],a.get_dataSet()[1])
end=time.time()

print(end-beg)