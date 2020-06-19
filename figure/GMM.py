import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-30, 30, 100)
mu = -15
sigma = 10
m = 15
sig = 8

y = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
y2 = 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-m)**2/(2*sig**2))
y3 = y + y2

plt.plot(x, y, 'r')
plt.plot(x, y2, 'b')
plt.plot(x, y3, 'y-')

plt.legend(["N1", "N2", "N"])

plt.show()
