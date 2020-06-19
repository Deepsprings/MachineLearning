import numpy as np
import matplotlib.pyplot as plt

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 663., 619., 393., 428., 27., 193., 66., 226., 1591.]
# y_data = b + w * x_data
#plt.xlabel("x_data")
#plt.ylabel("y_data")
#plt.scatter(x_data, y_data)
#plt.show()


b = -120	# initial b
w = -4	# initial w
#lr = 0.0000001	# learning rate
lr = 1
iteration = 100000



# store initial values for plotting
b_history = [b]
w_history = [w]

lr_b = 0
lr_w = 0

for i in range(iteration):
	b_grad = 0.0
	w_grad = 0.0
	for n in range(len(x_data)):
		b_grad = b_grad - 2.0*(y_data[n] - (b + w*x_data[n]))*1.0
		w_grad = w_grad - 2.0*(y_data[n] -(b + w*x_data[n]))*x_data[n]
	
	lr_b = lr_b + b_grad ** 2
	lr_w = lr_w + w_grad ** 2


	# update parameters
	b = b - lr/np.sqrt(lr_b) * b_grad
	w = w - lr/np.sqrt(lr_w) * w_grad

	# store parameters for plotting
	b_history.append(b)
	w_history.append(w)

# plot the figure
plt.plot([-188.4], [2.67], 'x', color='red')
plt.plot(b_history, w_history, 'o-', color='blue')
plt.show()
