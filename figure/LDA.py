import matplotlib.pyplot as plt
import numpy as np

def test1():
    x1 = np.random.uniform(-2, 7, 35)
    n1 = np.random.uniform(-2, 2, 35)
    y1 = 0.85*x1 -3 + n1

    x1_m = np.mean(x1)
    y1_m = np.mean(y1)

    x2 = np.random.uniform(0, 5, 15)
    n2 = np.random.uniform(-2, 2, 15)
    y2 = 0.6*x2 -7 + n2

    x2_m = np.mean(x2)
    y2_m = np.mean(y2)


    l1 = [6, -2]
    l2 = [-10, 6]


    plt.scatter(x1, y1, color="c")
    plt.scatter(x2, y2, color="b")

    plt.scatter(x1_m, y1_m, color="black", marker="x")
    plt.scatter(x2_m, y2_m, color="black", marker="x")

    plt.plot(l1, l2, color="red")

    plt.xlim(-4, 10)
    plt.ylim(-10, 6)

    plt.legend(['line', 'class 1', 'class 2','the mid point of class 1', 'the mid point of class 2'])

    plt.show()

def demo():
    x1 = np.random.uniform(-1, 3, 30)+5
    y1 = np.random.uniform(-1, 3, 30)+4
    m1 = np.array([np.mean(x1), np.mean(y1)])

    plt.scatter(x1, y1, color="purple")
    plt.scatter(m1[0], m1[1], marker="x", color="black")


    x2 = np.random.uniform(-1, 3, 30)+1
    y2 = np.random.uniform(-1, 3, 30)-1
    m2 = np.array([np.mean(x2), np.mean(y2)])
    
    plt.scatter(x2, y2, color="b")
    plt.scatter(m2[0], m2[1], marker="x", color="black")


    x3 = np.random.uniform(-2, 3, 40)+5
    y3 = np.random.uniform(-2, 3, 40)
    m3 = np.array([np.mean(x3), np.mean(y3)])

    plt.scatter(x3, y3, color="c")
    plt.scatter(m3[0], m3[1], marker="x", color="black")

    m = [(m1[0]*30+m2[0]*30+m3[0]*40)/100, (m1[1]*30+m2[1]*30+m3[1]*40)/100]
    plt.scatter(m[0], m[1], marker="o", color="r")
    plt.legend(["class 1","mid point m_1", "class 2", "mid point m_2", "class 3", "mid point m_3", "total mid point m"])


    res1 = (np.dot(np.expand_dims(m1-m2, axis=0).T, np.expand_dims(m1-m2, axis=0)) + np.dot(np.expand_dims(m1-m3, axis=0).T, np.expand_dims(m1-m3, axis=0)))/2
    print(res1)

    res2 = (np.dot(np.expand_dims(m1-m, axis=0).T, np.expand_dims(m1-m, axis=0)))
    print(res2)

    plt.plot([m1[0],m[0]],[m1[1],m[1]], color="black")
    plt.plot([m2[0],m[0]],[m2[1],m[1]], color="black")
    plt.plot([m3[0],m[0]],[m3[1],m[1]], color="black")



















    plt.show()
    
if __name__ == "__main__":
    demo()
