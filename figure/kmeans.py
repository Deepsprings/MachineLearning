# 实现输入原始数据data:list和初始聚类点数据kdata:list，可视化k-means算法过程

import numpy as np
import matplotlib.pyplot as plt
import imageio

class kmeans:
    
    def __init__(self, data, kdata):
        self.data = data    # 原始数据 
        self.kdata = kdata  # 聚类中心
        self.k = len(self.kdata)    # 聚类数
        self.it = 0 # 迭代次数
        self.clusterIni(0)

    def clusterIni(self, sig):
        self.cluster = {}   # 聚类
        for i in range(self.k):
            if i==0 and sig==0:
                self.cluster[i] = data
            else:
                self.cluster[i] = []

    # 迭代
    def iter(self):
        # 聚类
        self.clusterIni(1)
        for _i,i in enumerate(self.data):
            index = 0
            distance = 1.0
            for _j,j in enumerate(self.kdata):
                dis = (i[0]-j[0])**2 + (i[1]-j[1])**2
                if _j == 0:
                    distance = dis
                    index = 0
                elif dis <= distance:
                    distance = dis
                    index = _j
            self.cluster[index].append(i)

        # 计算新聚类中心
        count = 0
        for c in self.cluster:
            sx = 0
            sy = 0
            for d in self.cluster[c]:
                sx = sx + d[0]
                sy = sy + d[1]
            avgx = sx / len(self.cluster[c])
            avgy = sy / len(self.cluster[c])
            if avgx == self.kdata[c][0] and avgy == self.kdata[c][1]:
                count = count + 1
            else:
                self.kdata[c] = [avgx,avgy]
        self.it = self.it + 1
        if(count == len(self.kdata)):
            print("总迭代次数:%d"%(self.it-1))
            return 0
        else:
            return 1


    # 根据cluster可视化
    def plot(self, isSave = 0):
        # 显示簇
        for c in self.cluster:
            if(self.cluster[c] == []):
                continue
            x = np.array(self.cluster[c])[:,0]
            y = np.array(self.cluster[c])[:,1]
            plt.scatter(x, y)

        # 显示中心点
        mx = np.array(self.kdata)[:,0]
        my = np.array(self.kdata)[:,1]
        plt.scatter(mx, my, marker='x', color='black')

        plt.title("After %d iterator"%self.it)
        if isSave:
            plt.savefig("./cluster/c%d.png"%self.it)
        plt.show()

if __name__ == '__main__':

    # 原始数据data
    data = []
    f = open('./clusterData2.txt', 'r')
    for _d in f:
        dat = _d.rstrip().split(' ')
        data.append([float(dat[0]), float(dat[1])])
    f.close()

    # 初始聚类中心点kdata
    kdata = [[3,3],[6,5],[10,1]]

    obj = kmeans(data, kdata)
    # 迭代10次
    for i in range(20):
        obj.plot(1)
        b = obj.iter()
        # 或到稳定时结束
        if not b:
            print('迭代结束!')
            break
        #obj.plot(1)
    #obj.plot()

    # 合成成gif动画
    inp = []
    for i in range(obj.it):
        inp.append(imageio.imread('./cluster/c%d.png'%i))
    outp = './cluster/cluster2.gif'
    imageio.mimsave(outp, inp, duration=1)

