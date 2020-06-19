import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import queue

# 原始数据生成
def genData():
    theta = np.random.uniform(0, 2*np.pi, 250)
    r = np.random.uniform(4.5, 5.5, 250)
    x1 = r*np.sin(theta) + 5
    y1 = r*np.cos(theta) + 5
    x2 = np.random.uniform(1, 3, 20) + 1.5
    y2 = np.random.uniform(1, 3, 20) + 5
    x3 = np.random.uniform(1, 3, 20) + 4.5
    y3 = np.random.uniform(1, 2, 20) + 5
    x4 = np.random.uniform(4, 7, 30)
    y4 = np.random.uniform(2, 4, 30)
    x_ = np.random.uniform(0, 20, 50) - 5
    y_ = np.random.uniform(0, 20, 50) - 5
    x = np.hstack((x1, x2, x3, x4, x_)) + 5
    y = np.hstack((y1, y2, y3, y4, y_)) + 5
    plt.scatter(x, y)
    plt.xlim((0, 15))
    plt.xlim((0, 15))
    plt.axis('equal')
    plt.show()
    dataF = pd.DataFrame({'x':x, 'y':y})
    dataF.to_csv('./data.csv')

# 读取原始数据集
def getData():
    data_0 = pd.read_csv('./data.csv')
    dataA = np.array(data_0)[:,1:]
    dataL = dataA.tolist()
    # plt.scatter(dataA[:,0], dataA[:,1])
    # plt.show()
    return dataL



class DBSCAN:
    def __init__(self, data, eps, min_samples):
        self.data = data # 原始数据
        self.eps = eps
        self.minP = min_samples - 1
        self.core = []
        self.k = 0
        self.cluster = {} # 聚类结果
    
    # 得到某点 d<=r 的邻域点, data数据集，isCont是否包含自身
    def getN(self, point, r, isCont = 0):
        data = self.data[:]
        data.remove(point)
        Nei = []
        for i in data:
            dis = (i[0]-point[0])**2 + (i[1]-point[1])**2
            if dis <= r**2:
                Nei.append(i)
        if isCont:
            Nei.append(point)
        return Nei
    
    # 获取两点集的交集
    def intersection(self, ptL1, ptL2):
        inters = []
        for i in ptL1:
            if i in ptL2:
                inters.append(i)
        return inters
        
    # 获取核心对象, isPlot是否可视化
    def getCore(self, isPlot = 0):
        data = self.data[:]
        for i in data:
            temp = self.getN(i, self.eps)
            if len(temp) >= self.minP:
                self.core.append(i)
        if isPlot:
            x = np.array(self.core)[:,0]
            y = np.array(self.core)[:,1]
            plt.scatter(np.array(self.data)[:,0], np.array(self.data)[:,1])
            plt.scatter(x, y)
            plt.legend(('rawPoint','corePoint'))
            plt.show()
    
    # 算法迭代
    def dbscan(self):
        q = queue.Queue()
        core = self.core[:]
        Data = self.data[:]
        while len(core) != 0:
            dat = Data[:]
            q.put(core[0])
            Data.remove(core[0])
            while not q.empty():
                pt = q.get()
                Nei = self.getN(pt, self.eps)
                if len(Nei) >= self.minP:
                    delta = self.intersection(Nei, Data)
                    for i in delta:
                        q.put(i)
                        Data.remove(i)
            self.k = self.k + 1
            self.cluster[self.k] = dat[:]
            for j in Data:
                self.cluster[self.k].remove(j)
            for k in self.cluster[self.k]:
                try:
                    core.remove(k)
                except:
                    continue
        self.cluster[0] = Data   # key = 0,对应于噪点
    
    def plot(self):
        for i in self.cluster:
            x = np.array(self.cluster[i])[:,0]
            y = np.array(self.cluster[i])[:,1]
            plt.scatter(x, y)
        plt.show()

# 测试
if __name__ == '__main__':
    genData()
    dataL = getData()
    db = DBSCAN(dataL, 0.8, 5)
    db.getCore()
    db.dbscan()
    db.plot()


    # 使用官方库
    import sklearn.cluster as skc
    dataA = np.array(dataL)
    y_pred = skc.DBSCAN(eps = 0.8, min_samples = 5).fit_predict(dataA)
    plt.scatter(dataA[:, 0], dataA[:, 1], c=y_pred)
    plt.show()
