import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def PCA():
    x0 = np.array([[-4,-3,-2,-1,0,1,2,3,4,5],[-5,-4,-1,-3,0,3,3,4,3,7]], dtype=np.float)
    x = x0.copy()
    
    data = pd.DataFrame(x)
    print("原始数据: \n",data,"\n")
    
    # 中心化
    x[0] = x[0] - np.sum(x[0])/len(x[0])
    x[1] = x[1] - np.sum(x[1])/len(x[1])
    
    data2 = pd.DataFrame(x)
    print("中心化后: \n",data2,"\n")

    # 协方差矩阵
    Sigma = np.dot(x,x.T)

    data3 = pd.DataFrame(Sigma)
    print("协方差矩阵: \n",data3,"\n")

    # 特征值与特征向量
    s = np.linalg.eig(Sigma)    # 基于协方差矩阵求解
    Lambda = s[0]
    print("基于协方差\n特征值: \n", Lambda,"\n")
    w = s[1]
    print("特征向量: \n",w,"\n")

    U,S,v = np.linalg.svd(x) # 基于 SVD 分解求解
    print("基于SVD\nS: \n",S,"\n")
    print("U: \n",U,"\n")
    
    # 投影矩阵
    W = w[:,1]
    print("投影矩阵: \n",W,"\n")

    # PCA 轴
    xx = np.linspace(-6,6,7)
    y1 = W[1]/W[0] * xx
    y2 = w[:,0][1]/w[:,0][0] * xx
    plt.plot(xx,y1,color="r")
    plt.plot(xx,y2,color="c")
    plt.legend(["PCA1","PCA2"])
    
    # np.dot(w[:,1],w[:,0].T) = 0 # 轴是垂直的

    # 降维后的值
    xx = np.dot(W,x0)
    print("降维后值: \n",xx,"\n")


    plt.scatter(x[0],x[1])
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.grid()
    plt.show()
    
    k1 = w[:,1][1]/w[:,1][0]
    k2 = w[:,0][1]/w[:,0][0]
    return k1,k2

def LDA():
    x1 = np.array([[-4,-3,-2,-1,0],[-5,-4,-1,-3,0]], dtype=np.float)
    x2 = np.array([[1,2,3,4,5],[3,3,4,3,7]], dtype=np.float)
    l1 = len(x1[0])
    l2 = len(x2[0])
    l = l1 + l2
    p1 = l1/l   # 类1概率
    p2 = l2/l   # 类2概率
    m1 = np.array([[np.sum(x1[0,:])/len(x1[0,:])],[np.sum(x1[1,:])/len(x1[1,:])]])  # 类1中心（均值）
    m2 = np.array([[np.sum(x2[0,:])/len(x2[0,:])],[np.sum(x2[1,:])/len(x2[1,:])]])  # 类2中心（均值）
    m = p1*m1 + p2*m2   # 总体中心（均值）

    data1 = pd.DataFrame(x1)
    data2 = pd.DataFrame(x2)
    print("原始数据: \nClass 1:\n",data1,"\n")
    print("Class 2:\n",data2,"\n")

    # 类内散度矩阵
    Sw_1 = np.cov(x1) * (l1-1)/l
    Sw_2 = np.cov(x2) * (l2-1)/l
    Sw = Sw_1 + Sw_2
    print("Sw : \n",Sw,"\n")

    # 类间散度矩阵
    Sb_1 = l1 * np.dot(m1-m,(m1-m).T)/l
    Sb_2 = l2 * np.dot(m2-m,(m2-m).T)/l
    Sb = Sb_1 + Sb_2
    print("Sb : \n",Sb,"\n")

    # 总散度矩阵
    x0 = np.hstack((x1,x2))/l
    St = np.cov(x0) * (l-1)
    print("St : \n",St,"\n")
    print("Sw+Sb: \n",Sw+Sb,"\n")

    # 另一种求法
    x0[0] = x0[0] - np.sum(x0[0])/len(x0[0])
    x0[1] = x0[1] - np.sum(x0[1])/len(x0[1])
    St = np.dot(x0,x0.T)

    # 特征值与特征向量
    S = np.dot(np.linalg.inv(Sw),Sb)
    s = np.linalg.eig(S)
    Lambda = s[0]
    w = s[1]
    print("特征值 : \n",Lambda,"\n")
    print("特征向量 : \n",w,"\n")

    # 投影矩阵
    W = w[:,1]
    print("投影矩阵 : \n",W,"\n")

    # LDA 轴
    xx = np.linspace(-6,6)
    y1 = W[1]/W[0] * xx
    y2 = w[:,0][1]/w[:,0][0] * xx
    plt.plot(xx,y1,color="r")
    plt.plot(xx,y2,color="c")
    plt.legend(["LDA1","LDA2"])

    # np.dot(w[:,1],w[:,0].T) != 0  轴不一定垂直

    # 降维后的值
    xx = np.dot(W,x0)
    print("降维后值: \n",xx,"\n")

    plt.scatter(x1[0],x1[1],color="b")
    #plt.scatter(m1[0],m1[1],color="b",marker="x")
    plt.scatter(x2[0],x2[1],color="r")
    #plt.scatter(m2[0],m2[1],color="r",marker="x")
    #plt.scatter(m[0],m[1],color="black",marker="x")
    plt.xlim(-8,8)
    plt.ylim(-8,8)
    plt.grid()
    plt.show()

    k1 = w[:,1][1]/w[:,1][0]
    k2 = w[:,0][1]/w[:,0][0]
    return k1,k2


#k1,k2 = PCA()
k3,k4 = LDA()
'''
x1 = np.array([[-4,-3,-2,-1,0],[-5,-4,-1,-3,0]], dtype=np.float)
x2 = np.array([[1,2,3,4,5],[3,3,4,3,7]], dtype=np.float)
 

plt.scatter(x1[0],x1[1],color="b")
#plt.scatter(m1[0],m1[1],color="b",marker="x")
plt.scatter(x2[0],x2[1],color="r")

xx = np.linspace(-6,6)
y1 = k1 * xx
y2 = k2 * xx
y3 = k3 * xx
y4 = k4 * xx
plt.plot(xx,y1,"r")
plt.plot(xx,y2,"r:")
plt.plot(xx,y3,"c")
plt.plot(xx,y4,"c:")
plt.legend(["PCA1","PCA2","LDA1","LDA2"])
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.grid()
plt.show()

'''
