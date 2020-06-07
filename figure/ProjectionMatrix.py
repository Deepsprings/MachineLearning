import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)

x = [1,5]
y = [1,5]
z = [0,4]

x1 = [1,2]
y1 = [1,1]
z1 = [0,0]

x2 = [1,1]
y2 = [1,2]
z2 = [0,0]

x3 = [1,5]
y3 = [1,5]
z3 = [0,0]

x4 = [5,5]
y4 = [5,5]
z4 = [0,4]



ax.plot3D(x,y,z,color="r")
ax.plot3D(x1,y1,z1,color="c")
ax.plot3D(x2,y2,z2,color="c")
ax.plot3D(x3,y3,z3,color="pink")
ax.plot3D(x4,y4,z4,color="black",marker=".")
plt.show()
