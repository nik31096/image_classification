import numpy as np

class Network:
	def __init__(self, layers, alpha=0.1):
		self.W = []
		self.layers = layers
		self.alpha = alpha
		# filling W - list of matrix, each corresponding to certain layer in NN
		# +1 in w matrix dimention is responsible for including the bias column in itself
		for i in np.arange(0, len(layers) - 2):
			w = np.random.randn(layers[i] + 1, layers[i+1] + 1)
			self.W.append(w/np.sqrt(layers[i]))
		# but the last layer doesn't need the bias, so matrix for last two layers looks like this
		w = np.random.randn(layers[-2] + 1, layers[-1])
		self.W.append(w/np.sqrt(layers[-2]))
		
	def __repr__(self):
		return "Neural Network: {}".format(', '.join([str(i) for i in self.layers]))
	
	def sigmoid(self, x):
		return 1.0/(1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		return x*(1-x)
		
	def train(self, X, y, epochs=1000, ep2show=100):
		# including bias column in X
		X = np.c_[X, np.ones((X.shape[0]))]
		
		for epoch in np.arange(0, epochs):
			for (x, target) in zip(X, y):
				# heart of backprobagation
				self.train_partial(x, target)
				
			if epoch == 0 or (epoch % ep2show) == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch: {}, loss: {:.7f}".format(epoch, loss))
				
	def train_partial(self, x, y):
		# list, storing output activations for each layer as our data point propagate though NN
		A = [np.atleast_2d(x)]
		# starting forward propagation phase
		for layer in np.arange(0, len(self.W)):
			net = A[layer].dot(self.W[layer])
			out = self.sigmoid(net)
			A.append(out)
		# starting back propagation phase
		# calculating difference between NN output, i.e. A[-1] and y 
		error = A[-1] - y
		# is used to update weight matrix NN
		D = [error*self.sigmoid_deriv(A[-1])]
		for layer in np.arange(len(A) - 2, 0, -1):
			delta = D[-1].dot(self.W[layer].T)
			d = delta* self.sigmoid_deriv(A[layer])
			D.append(d)
		D = D[::-1]
		# weight update phase
		for layer in np.arange(0, len(self.W)):
			self.W[layer] += -self.alpha*A[layer].T.dot(D[layer])
			
	def predict(self, X, addBias=True):
		# output vector, init with X
		p = np.atleast_2d(X)
		if addBias:
			p = np.c_[p,  np.ones((p.shape[0]))]
			
		for layer in np.arange(0, len(self.W)):
			p = self.sigmoid(np.dot(p, self.W[layer]))
		return p
	
	def calculate_loss(self, X, targets):
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias= False)
		loss = 0.5*np.sum((targets - predictions)**2)
		
		return loss

if __name__ == "__main__":
	X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
	y =  np.array([[0], [1], [1], [0]])

	nn = Network([2, 2, 1], alpha=0.1)
	nn.train(X, y, epochs=20000)

	for (x, target) in zip(X, y):
		pred = nn.predict(x)[0][0]
		step = 1 if pred>0 else 0
		print("[INFO] data={}, true-value={}, pred={:.4f}, step={}".format(x, target, pred, step))