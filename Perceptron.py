import numpy as np
import matplotlib.pyplot as plt

class perceptron:
	def __init__(self, x, y, r):
		self.x = np.array(x)
		self.y = np.array(y)
		self.w = np.zeros(shape = (1, self.x.shape[-1]))
		self.b = 0
		self.r = r

	def fit(self):
		sig = 1
		while(sig):
			for i,x_data in enumerate(self.x):
				if(self.y[i] * (np.dot(self.w, x_data.T) + self.b) <=0):
					self.w = self.w + self.r * self.y[i] * x_data
					self.b = self.b + self.r * self.y[i]
					break
				elif( i == len(self.x)-1):
					sig = 0

	def model(self):
		# print("w:", self.w)
		# print("b:", self.b)
		return self.w,self.b

def main():
	x_data = [[3,3], [4,3], [5,2], [1,1], [1.1, 1.1], [-1,0], [4,4]]
	y_label = [1, 1, 1, -1, 1, -1, 1]
	learning_rate = 1
	
	fig = plt.figure()
	plt.plot(np.array(x_data)[:, 0], np.array(x_data)[:, 1], 'o')
	#plt.show()

	p = perceptron(x_data, y_label, learning_rate)
	p.fit()
	w,b = p.model()
	print("w:",w)
	print("b:",b)

	x = np.arange(5)
	y = -w[0][0]/float(w[0][1]) * x - b/float(w[0][1])
	plt.plot(x, y)
	plt.show()

if __name__ == '__main__':
	main()
