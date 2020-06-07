import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-0, 2, 100)
y1 = np.sqrt(2*x-x**2)
y2 = -y1

np.random.seed(1)
x0 = np.random.rand(180) * 4 - 1
y0 = np.random.rand(180) * 4 - 2
z11 = x0**2
z22 = y0**2
s1,s2 = np.meshgrid(z11, z22)
s3 = 0.5 * (s1 + s2)

fig = plt.figure()
ax = Axes3D(fig)

ax.plot_surface(s1, s2, s3, rstride=1, cstride=1, color='grey')
for i in list(zip(x0, y0)):
    z1 = i[0]**2
    z2 = i[1]**2
    z3 = i[0]
    if(i[1]**2 > 2*i[0] - i[0]**2):
        ax.scatter(z1, z2, z3, color='red')
        pass
        #plt.scatter(i[0], i[1], marker='+', color='red')
        
    elif(i[1]**2 < 2*i[0] - i[0]**2):
        ax.scatter(z1, z2, z3, color='blue')
        pass
        #plt.scatter(i[0], i[1], marker='.', color='blue')

#plt.plot(x, y1, '--', color = 'silver')
#plt.plot(x, y2, '--', color = 'silver')

plt.show()
