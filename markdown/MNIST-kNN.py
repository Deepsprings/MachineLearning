from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def	loadData():
	
	# 读取mnist手写数字集数据
	mnist_dir = '/home/zhangwei/data/database/MNIST'
	mnist = input_data.read_data_sets(mnist_dir, one_hot = True)
	# 图片数据
	train_images = mnist.train.images	# (55000, 784)
	val_images = mnist.validation.images	# (5000, 784)
	test_images = mnist.test.images		# (10000, 784)
	# 标签信息
	train_labels = mnist.train.labels	# (55000, 10)
	val_labels = mnist.validation.labels
	test_labels = mnist.test.labels
	
	return train_images,train_labels,test_images,test_labels

def kNN(train_images,train_labels,test_images,test_labels, k = 10, sample_size = 55000):
	# k:近邻系数

	# 截取部分数据进行kNN测试
	train_img = train_images[0:sample_size, :]
	train_label = train_labels[0:sample_size, :]
	test_img = test_images
	test_label = test_labels
	
	# 使用逐行扫描的方法,使用多数表决策略
	rNum = 0
	wNum = 0
	batch = 100 # 测试100个test_img，查看准确率
	for i in range(batch):
		Diff = np.tile(test_img[i], (train_img.shape[0], 1)) - train_img
		sqDiff = Diff ** 2
		sqDistance = sqDiff.sum(axis = 1)
		d = sqDistance ** 0.5
		index = d.argsort()	# 返回排序后的索引

		classList = np.zeros((1, 10))
		for j in index[0:k]:
			classList = train_label[j] + classList

		predict = classList.argsort()[0,-1]
		fact = test_label[i].argsort()[-1]
		if(predict == fact):
			rNum = rNum + 1
		else:
			wNum = wNum + 1
	
	accuracy = rNum / float(wNum + rNum)
	return accuracy

if __name__ == '__main__':

	train_images,train_labels,test_images,test_labels = loadData()
	# k取不同值时，计算精度acc
	k = np.arange(1, 21)
	acc = []
	for n,i in enumerate(k):
		acc.append(kNN(train_images,train_labels,test_images,test_labels, i,10000))
		print(n, '/', len(k))
	plt.plot(k, acc, 'o')
	plt.show()
