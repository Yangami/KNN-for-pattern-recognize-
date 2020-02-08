# coding:utf-8
from __future__ import division
import pandas as pd
import numpy as np
from kuanke.user_space_api import *
from jqdata import *
try:
    import configparser
except:
    import ConfigParser as configparser
import sys
import io

# 初始化函数，设定基准等等


class pattern:
    def __init__(self):
        self = self

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

        if 'tp' in list(classCount.keys()):
            if classCount['tp'] == np.max(list(classCount.values())):
                maxIndex = 'tp'
            else:
                maxIndex = 'fp'
        else:
            maxIndex = 'fp'
        return maxIndex, classCount

    def parse_args(self,filename):
        txt = read_file('cfg.txt')
        cf = configparser.RawConfigParser(allow_no_value=True)
        cf.readfp(io.BytesIO(txt))
        # 查看所有节点
        secs = cf.sections()
        print(secs)
        # init section
        init = cf.items("init")
        dic = {}
        for k, v in init:
            dic[k] = v
        # 从 init 中获取path
        path = dic['path']
        # 从特定形态section中获取文件名
        files = cf.items(dic['cur_pattern'])
        t_file = files[0][1]
        f_file = files[1][1]
        # read
        dirs = [path + t_file, path + f_file]

        return dirs
    def get_W_dataSet(self):
        axx, labels = [], []
        #dir = config.cfg()  # ['t_parameter.txt', 'f_parameter.txt','t_parameter13.txt', 'f_parameter16.txt']
        #lines = read_file('cfg.txt').split('\n')
        #print(lines)
        dir=self.parse_args('cfg.txt')
        lab_dic = {'t': 'tp', 'f': 'fp'}
        for i in dir:
            print(i)
            if 'b' in [str(read_file(i))[0], str(read_file(i))[1]]:  # ,'/home/jquser/'+
                lines = str(read_file(i))[2:-1].split('\\n')
            else:
                lines = read_file(i).split('\n')
            f = []
            for line in lines:
                if len(line) != 0:
                    axx.append([float(j) for j in line.split(',')])
                    labels.append(lab_dic[i[i.find('parameter') - 2]])
            dataSet, labels = np.array(axx), labels  # createDataSet()
        return dataSet, labels

    def get_rawdata(self, security, end_date, start_num, end_num, step, unit='1d', include_now=True, fq_ref_date=None):
        closeframe = []
        if type(security) == str:
            security = [security]
        for st in security:
            closelst = []
            for num in range(start_num, end_num, step):
                df = get_bars(str(st), count=num, unit=unit, fields='close', end_dt=end_date, include_now=include_now,
                              fq_ref_date=fq_ref_date)
                close = df['close']
                if len(close) == 0:
                    print(st + '_data_wrong')
                    break
                closelst.append(close)
            if len(closelst) != end_num - start_num:
                print('wrong_data', len(closelst), end_num - start_num, closelst)
                break
            closeframe.append(closelst)

        return closeframe

    def get_index_rawdata(self, sectors, end_date, start_num, end_num, step, frequency='1d'):
        closeframe = []
        for sector in sectors:
            closelst = []
            closedata = self.get_index_price(name=sector, end_date=end_date, count=end_num)
            for num in range(end_num, start_num, -1 * step):
                if len(closedata) == 0:
                    # print(sector+'_data_wrong')
                    close = [0] * num
                else:
                    close = closedata[-1 * num:]
                closelst.append(close)
            if len(closelst) != end_num - start_num:
                print('wrong_data', sector, len(closelst), end_num - start_num, closelst)
                pass
            else:
                closeframe.append(closelst)
        return closeframe

    def datastd(self, rawdata, compress_size=30):
        print('stding')
        stock_df = pd.DataFrame()
        testdata = []
        diff = []
        for closelst in rawdata:
            test_x = []
            dif = []
            for close in closelst:
                x = []
                df = pd.DataFrame({'close': close}, index=[i for i in range(len(close))])
                g = int(len(df['close']) * 0.15)
                dif.append(( df['close'][0]- np.min(df['close'][g:-g])) /( np.max(df['close'][g:-g])- np.min(df['close'][g:-g])))
                if len(df) == 0:
                    print('df=0')
                    continue
                df['close'] = (df['close'] - np.min(df['close'])) / (np.max(df['close']) - np.min(df['close']))
                x.append(df['close'][0])
                for i in range(compress_size):
                    x.append(df['close'][max(int((i + 1) * (len(df) / compress_size)) - 1, 0)])
                test_x.append(list(x))

            testdata.append(test_x)
            diff.append(dif)
        return testdata, diff

    def dw(self, rawdata, stocklst):
        print('caculating')
        dataSet, labels = self.get_W_dataSet()
        itestdata, dif = self.datastd(rawdata)
        resultlst = []
        position = []
        # if stocklst[0][-4:-1]!='XSH':
        #   stocklst=[i for i in stocklst if get_industry_stocks(i)!=[]]
        for j, i in enumerate(itestdata):
            # print(j,np.shape(np.array(itestdata)))
            testX = i
            pred = []
            post = -1
            for idx, test_i in enumerate(testX):
                # print(np.shape(np.array(test_i)))
                outputLabel, tnum = self.kNNClassify(test_i, dataSet, labels)
                if outputLabel == 'tp':
                    if max(test_i[-5:]) / (max(test_i[5:]) - min(test_i[5:])) < 0.45:
                        pred.append(1)
                        post = idx
                        break
                    # 写文件
                else:
                    pred.append(0)
            resultlst.append(int(np.sum(pred)))
            position.append(post)

        wlst = [stocklst[i] for i, j in enumerate(resultlst) if j != 0]
        diff = [dif[i][position[i]] for i, j in enumerate(resultlst) if j != 0]

        st_dict = {}
        for i, j in enumerate(wlst):
            st_dict[j] = diff[i]
        return st_dict

    def record(self, filename, result):

        write_file(str(filename) + '.txt', '股票代码,价差' + '\n', 'a')

        for i in result:
            write_file(str(filename) + '.txt', str(i) + ',' + str(result[i]) + '\n', 'a')

    def get_index_price(self, name, end_date, count, start_time=None, frequency='1d'):
        industry_list1 = get_industries(name='sw_l1').index
        industry_list1 = list(industry_list1)
        # 2.行业(申万2级行业，共110个)
        industry_list2 = get_industries(name='sw_l2').index
        industry_list2 = list(industry_list2)
        industry_list3 = list(get_industries(name='sw_l3').index)
        # 3.概念(共729个)
        concept_list = get_concepts()
        concept_list1 = list(concept_list.index)
        if name in industry_list1 or name in industry_list2:
            # start_time = '2015-01-01'
            stock_list = get_industry_stocks(name, date='2019-01-01')
        if name in concept_list1:
            start_time = concept_list['start_date'][name]
            stock_list = get_concept_stocks(name, date='2019-01-01')
        if name in industry_list3:
            stock_list = get_industry_stocks(name, date='2019-01-01')
        if len(stock_list) == 0:
            # print('no stock in', name)
            return []
        now = end_date
        index = []
        price = get_price(security=stock_list, end_date=now, count=count, start_date=None, frequency='1d',
                          fields=['close'])
        for stock in stock_list:
            normal = price['close'][stock] / price['close'][stock][0]
            if len(index) == 0:
                index = normal
            else:
                index = index + normal
        # 标准化
        index = index * 100 / len(stock_list)
        return list(index)
