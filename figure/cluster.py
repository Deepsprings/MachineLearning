import numpy as np
import matplotlib.pyplot as plt

def fig1():
    p1 = [1,1]
    p2 = [7,2]
    p3 = [3,6]

    np.random.seed(1)
    x1 = np.random.uniform(-2,2,30) + p1[0]
    y1 = np.random.uniform(-2,1,30) + p1[1]
    plt.scatter(x1, y1)

    x2 = np.random.uniform(-2,2,30) + p2[0]
    y2 = np.random.uniform(-2,2,30) + p2[1]
    plt.scatter(x2, y2)

    x3 = np.random.uniform(-2,2,30) + p3[0]
    y3 = np.random.uniform(-2,2,30) + p3[1]
    plt.scatter(x3, y3)

    x = np.hstack((x1, x2, x3))
    y = np.hstack((y1, y2, y3))

    return np.dstack((x, y))

    #plt.legend(["cluster 1","cluster 2","cluster 3"])
    #plt.show()

def fig2():

    np.random.seed(2)
    x1 = np.linspace(2,10, 50)
    y1 = np.sqrt(25 - (x1-6)**2)  + np.random.uniform(0,1,50)

    x2 = np.linspace(4,12,50)
    y2 = 4 - np.sqrt(25 - (x2-8)**2) + np.random.uniform(-1,0,50)


    plt.scatter(x2+2, y2+4)
    plt.scatter(x1, y1)


    plt.legend(["cluster 1","cluster 2"])
    plt.show()


def fig3():
    pass


class kmeans:
    def __init__(self, data, kdata):
        self.data = data
        self.kdata = kdata
        self.distance = np.zeros((data.shape[0], kdata.shape[0]),dtype = np.float)
        self.cluster = np.empty((kdata.shape[0], 1))

        plt.scatter(data[:,0],data[:,1],color = "c")
        plt.scatter(kdata[:,0],kdata[:,1], color = "r")
        plt.show()

    def iter(self):
        for _i,i in enumerate(data):
            for _j,j in enumerate(kdata):
                dis = np.sqrt((i[0]-j[0])**2 + (i[1]-j[1])**2)
                self.distance[_i,_j] = dis

            index = np.argmax(self.distance[_i])
            print(i)
            print(self.cluster)
            print(np.insert(self.cluster[index], -1, i, 0))
            print(self.cluster)




    def go(self):
        self.iter()


if __name__ == "__main__":
    data = fig1()[0]
    kdata = np.array([[1,1],[3,4],[7,2]])
    km = kmeans(data,kdata)
    km.go()

