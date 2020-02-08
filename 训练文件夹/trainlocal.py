import pandas as pd
import winsound
import numpy as np
import configparser
import datetime
import time
import copy
import os
beg=time.time()
# KNN部分函数定义
class train():
    def _init_(self):
        pass
    def parse_args(self,filename):
        cf = configparser.ConfigParser()
        cf.read(filename)
        init = cf.items("init")
        dic = {}
        epoch = int(dict(cf.items('init'))['epoch'])
        for k, v in init:
            dic[k] = v
        folw=dic['follow']
        modelpath = dic[dic['model_path']]
        rformat=dic['rdata_format']
        rdata = cf.items(str(rformat))
        # 从 init 中获取path
        if folw:
            assert 'model/model'in dic[dic['model_path']]

            modelpath =[modelpath[:modelpath.rindex('/model')+6]+str(epoch)+'/'+cf.items('model')[0][1],modelpath[:modelpath.rindex('/model')+6]+str(epoch)+'/'+cf.items('model')[1][1]]
        trainpath=dic['train_path']
        rdatapath=dic['rdata_path']
        files = dict(cf.items('train'))#训练文件
        t_file,f_file=files['true'],files['false']
        tpath,fpath=files['tpath'],files['fpath']
        addnum=int(files['add_num'])
        if files['follow']:
            addnum=epoch
        addpath=[]
        for num in range(addnum):
            addpath.append(tpath+str(num+1)+'.txt')
            addpath.append(fpath+str(num+1)+'.txt')
        val=cf.items('valid')[0][1]#验证文件
        if rformat=='csv':
            rdata = [rdatapath+i[1] for i in rdata]
        elif rformat=='txt':
            rdata=rdata[0][1]
        k=int(dict(cf.items('init'))['k'])
        # read
        dirs = [trainpath + t_file, trainpath + f_file,trainpath+val,rformat,rdata,modelpath,addpath,epoch,k]
        return dirs
    def date_adj(self,date,idx,typ):
        if date in list(idx):
            pass
        else:
            date=self.date_adj(str(datetime.datetime.strptime(date, '%Y-%m-%d')+ datetime.timedelta(days = typ))[:10],idx,typ)
        return date
    def get_data(self,df,security, start_date = '0', end_date='0', count='0'):
        try:
            close=df[security]
        except:
            return []
        # except:
        #     print('5454')
        #     return []
        if (end_date not in close.index)or (start_date not in close.index):
            return []
        if end_date in list(close.index):
            pass
        else:
            end_date=str(datetime.datetime.strptime(end_date, '%Y-%m-%d')+ datetime.timedelta(days = -1))[:10]

        if count=='0':
            stidx=list(close.index).index(start_date)
            edidx=list(close.index).index(end_date)
            closelst=list(close)[stidx:edidx]
            closelst.append(list(close)[edidx])
            return closelst
        else:
            try:

                edidx=list(close.index).index(end_date)
                stidx =edidx+count
                close=close[stidx:edidx]
                close.append(close[edidx])
                return close
            except:
                return []

    def createDataSet(self):
        # 建立构建分类器的原始样本
        data_dic={'x'}
        group = np.array(data_dic['x'])
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

        if '正样本' in list(classCount.keys()):
            if classCount['正样本'] == np.max(list(classCount.values())):
                maxIndex = '正样本'
            else:
                maxIndex = '负样本'
        else:
            maxIndex = '负样本'
        return maxIndex, classCount
    #获取训练集
    def get_dataSet(self,compress_size=30 ):

        (truedir, falsedir, dirs, rformat,rfilelst, _, addpath, _, k) = self.parse_args('cfg.txt')
        print('添加样本：' + str(len(addpath) / 2) + '个')
        axx, labels = [], []
        if rformat=='csv':

            rdf = pd.read_csv(rfilelst[0], index_col=0)
            for i in rfilelst[1:]:
                df1 = pd.read_csv(i, index_col=0)
                rdf = pd.concat([rdf, df1], axis=1)
        elif rformat=='txt':
            dic={}

            for dir in os.listdir(rfilelst):
                dic[dir[:-4]]=pd.read_csv(rfilelst+dir,sep='\t',index_col=0)
            pl = pd.Panel(dic)

            rdf=pl.minor_xs('close')
        dir = [truedir, falsedir]
        lab_dic = {truedir: '正样本',falsedir: '负样本'}
        for name in dir:
            #print('read', name)
            f = open(name,encoding='utf-8').readlines()

            for j, i in enumerate(f[:-1]):
                if rformat=='csv':
                    if i[0]=='6':
                        security = i[0:6] + '.XSHG'
                    else:
                        security = i[0:6] + '.XSHE'
                else:
                    security=i[0:6]
                start_date = i[7:17]
                end_date = i[18:28]
                x = []
                close = self.get_data(rdf,security=security, start_date=start_date,end_date=end_date)
                if len(close)!=0:
                    df = pd.DataFrame({'close': close}, index=[i for i in range(len(close))])
                    df = df.dropna()
                if len(df) == 0:
                    continue
                df['close'] = (df['close'] - np.min(df['close'])) / (np.max(df['close']) - np.min(df['close']))
                x.append(df['close'][0])
                for i in range(compress_size):
                    x.append(df['close'][max(int((i + 1) * (len(df) / compress_size)) - 1, 0)])
                axx.append(list(x))
                labels.append(lab_dic[name])
        dataSet, labels = np.array(axx), labels  # createDataSet()
        print('原始训练集数量'+str(len(labels)))
        print(addpath)
        rlabel=copy.deepcopy(labels)
        #添加新生成的样本
        alab_dic = {'t': '正样本', 'f': '负样本'}
        for path in addpath:
            rds=open(path).readlines()
            result = [list(map(float, r.split(','))) for r in rds]
            try:
                np.array(result).reshape(-1,31)
            except:
                print('新样本长度不匹配')
            dataSet=np.concatenate((dataSet,result))
            for _ in range(len(result)):
                labels.append(alab_dic[path[6]])
        print('新添训练集数量' + str(len(labels)-len(rlabel)))
        print('训练集总数' + str(len(labels)))
        return dataSet, labels


# 获取W型股票列表,dirs格式security startdate enddate
    def get_wst(self,compress_size,k):
        pred = []
        (truedir, falsedir, dirs, rformat,rfilelst,_,_,_,_) = self.parse_args('cfg.txt')

        dataSet,labels=self.get_dataSet()
        if rformat=='csv':

            rdf = pd.read_csv(rfilelst[0], index_col=0)
            for i in rfilelst[1:]:
                df1 = pd.read_csv(i, index_col=0)
                rdf = pd.concat([rdf, df1], axis=1)
        f = open(dirs,encoding='utf-8').readlines()
        labelst=[]
        for j, i in enumerate(f[:]):
            test_x=[]
            if rformat=='csv':

                if i[0]=='6':
                    security = i[:6] + '.XSHG'
                else:
                    security = i[:6] + '.XSHE'
            elif rformat=='txt':
                security=i[:6]
            start_date = i[7:17]
            end_date = i[18:28]
            labelst.append(i[29:30])
            x = []
            if rformat=='txt':
                if i[:6]+'.txt' in os.listdir(rfilelst):
                    rdf =pd.DataFrame({i[:6]:pd.read_csv(rfilelst+i[:6]+'.txt','\t',index_col=0)['close']},columns=[i[:6]])
                else:
                    print(str(i[:6])+'数据不存在！')
                    continue
            close = self.get_data(rdf,security=security, start_date=start_date,end_date=end_date)
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
                print(str(i[:6])+'有缺失值')
            for j, i in enumerate(test_x):
                testX = i
                outputLabel, tnum = self.kNNClassify(testX, dataSet, labels, k)
                if outputLabel[0] == '正':
                    pred.append(1)
                    #print( " classified to class: ",outputLabel)
                else:
                    pred.append(0)
                    #print( " classified to class: ",outputLabel)
        corr=0
        for i,j in enumerate(pred):
            if (labelst[i]=='T')&(j==1):
                corr+=1
            if (labelst[i]=='F')&(j==0):
                corr+=1
        acc=corr/len(pred)
        return acc

    def select_k(self,sk):
        acclst = []
        best_k = []
        for i in range(5, 40):
            acc = train().get_wst(30, i)
            acclst.append((i, acc))
            print(i, acc)
            with open('select_k.txt', 'a') as u:
                u.write('k=' + str(i) + ',acc=' + str(train().get_wst(30, i)) + '\n')
        for i in acclst:
            if i[1] == max([j[1] for j in acclst]):
                best_k.append(i[0])
        with open('select_k.txt', 'a') as u:
            u.write('best_k=' + str(best_k))

    def get_model(self,dataSet,labels):
        (_,_,_,_,_,path,_,_,k) = self.parse_args('cfg.txt')
        # 判断结果
        for p in path:
            if os.path.exists(p):
                os.remove(p)
            p=p[:p.rindex('/')]
            if not os.path.exists(p):
                os.makedirs(p)
                print(p + ' 创建成功')
            else:
                print(p + ' 目录已存在')



        for i,j in enumerate(labels):
            if j=='负样本':
                with open(str(path[0]),'a') as u:
                    u.write(','.join([str(k) for k in dataSet[i]])+'\n')
                u.close()
            else:
                with open(str(path[1]),'a') as u:
                    u.write(','.join([str(k) for k in dataSet[i]])+'\n')
                u.close()

if __name__ == "__main__":
    a=train()
    (truedir, falsedir, dirs,rformat, rfilelst, modelpath, addpath,epoch,k) = a.parse_args('cfg.txt')
    #for k in range(35):
    acc=a.get_wst(30,k)


    print(k,'acc',acc)
    with open('train_log.txt','a') as f:
        f.write('epoch'+str(epoch)+',acc'+str(acc)+'\n')
    f.close()
    a.get_model(a.get_dataSet()[0],a.get_dataSet()[1])
    end=time.time()
    print('cost time:'+str(end-beg))
    #
    winsound.Beep(400, 1000)
