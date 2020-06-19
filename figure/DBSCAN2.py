# 实现输入原始数据data:list，阈值(半径)r,最少数据点数mp,可视化DBSCAN算法过程

import numpy as np
import matplotlib.pyplot as plt
import queue
import sklearn.cluster as skc

class dbscan:
    def __init__(self, data, r, mp):
        self.data = data
        self.r = r
        self.mp = mp
        self.k = 0
        
        self.clusterIni()
        self.core = []

    def clusterIni(self):
        self.cluster = {}
        self.cluster[self.k] = self.data

    # 获取核心对象集合(应该可以优化)
    def getCoreObj(self):
        for i in self.data:
            dis = 0.0
            count = -1
            for j in self.data:
                dis = (i[0]-j[0])**2 + (i[1]-j[1])**2
                if(dis <= self.r**2):
                    count = count + 1
            if count >= self.mp:
                self.core.append(i)
        self.plotCore()
    # 
    def getCluster(self):
        # 从核心对象中抽取数据
        Data = self.data[:]
        while len(self.core) != 0:
            Data_old = Data[:]
            core = self.core[0]
            q = queue.Queue()
            q.put(core)
            Data.remove(core)
            while not q.empty():
                temp = q.get()
                temp_l = []
                for j in Data:
                    dis = (temp[0]-j[0])**2 + (temp[1]-j[1])**2
                    if dis <= self.r**2 and dis != 0:
                        temp_l.append(j)
                if len(temp_l) >= self.mp:
                    x = np.array(Data)[:,0]
                    y = np.array(Data)[:,1]
                    for k1 in temp_l:
                        if (k1[0] in x) and (k1[1] in y):
                            q.put(k1)
                            Data.remove(k1)

            for d in Data:
                Data_old.remove(d)
            self.k = self.k + 1
            self.cluster[self.k] = Data_old
            for dat in Data_old:
                try:
                    self.core.remove(dat)
                except:
                    print(11111111)
        #print(self.cluster)


    def plotCore(self):
        plt.scatter(np.array(self.data)[:,0], np.array(self.data)[:,1])
        plt.scatter(np.array(self.core)[:,0], np.array(self.core)[:,1], marker="x")
        plt.show()


    def plotCluster(self):
        for i in self.cluster:
            if i == 0:
                continue
            x = np.array(self.cluster[i])[:,0]
            y = np.array(self.cluster[i])[:,1]
            plt.scatter(x, y)
        plt.show()

    
if __name__ == '__main__':
    # 原始数据data
    data = []
    f = open('./clusterData2.txt', 'r')
    for _d in f:
        dat = _d.rstrip().split(' ')
        data.append([float(dat[0]), float(dat[1])])
    f.close()

    obj = dbscan(data, 0.5, 3)
    obj.getCoreObj()
    obj.getCluster()
    print("簇个数：%d"%obj.k)
    obj.plotCluster()
    
    # 使用官方库
    db = skc.DBSCAN(eps=0.5, min_samples=3).fit(np.array(data))
    print(db.labels_)

