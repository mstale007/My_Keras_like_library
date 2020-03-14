class NeuralNetwork():
	
	def __init__(self):
		self.weights=[]
		self.layers=[]



	def add(self,shape,input_dim=0):
		if(input_dim==0):
			input_dim=self.layers[-1]
			self.weights.append(2*np.random.random((input_dim,shape))-1)
			self.layers.append(shape)
		else:
			self.layers.append(input_dim)

	def nonlin(self,x,deriv=False):
		#Sigmoid
		if(deriv==True):
			return np.exp(-x)/((1+np.exp(-x))**2)
		return 1/(1+np.exp(-x))

		#ReLU
		"""if(deriv==True):
				if(x>0):
					return 1
				else:
					return 0
			else:
				return max(x,0)"""

	def plot(self,images,cols=1,titles=None):
		import matplotlib.pyplot as plt
		import matplotlib.image as mpimg

		fig=plt.figure(figsize=(28,28))
		plt.gray()
		for i in range(10):
			plt.subplot(2,5,i+1)
			#fig.add_subplot(i+1,2,1)
			#plt.title("Weights %d" %(i+1))
			plt.imshow(im[:,i].reshape((28,-1)).astype('float32'))
		plt.show()


	def fit(self,X,Y,epochs=10,lr=0.001,lamda=0):
		layer_error=0
		#print("Initial weights:",self.weights[0])
		cost=[]
		for i in range(epochs):

			#Feed Forward
			forward=X
			prediction=self.predict(forward)

			error=prediction-Y
			forward=self.feed_forward(forward)
			#from sklearn.metrics import coverage_error
			#print("Error: ",coverage_error(Y,forward[-1]))
			cost.append(sum(error[0]**2)/2)
			print("Cost: ",cost[i])
			#layer_error will propogate backwards, itll keep on changin to the value of error at given layer in the loop
			layer_error=error

			#Will save all changes of self.weights to implement it simultaneously at the end
			dJdW={}
			delta=-layer_error
			for i in reversed(range(0,len(self.layers)-1)):
				#For appending from front side
				dJdW[i]=np.dot(forward['a'][i].T,delta)
				#print(forward['a'][i].shape,delta.shape,dJdW[i].shape)
				delta= np.multiply(np.dot(delta,self.weights[i].T), self.nonlin(forward['z'][i],True))
				
				#delta=np.dot(delta,self.nonlin(forward[i],True))
			#print(dJdW[0][0])

			#Changes to self.weights
			for i in range(0,len(self.weights)):
				#self.weights[-i]+=np.array(forward[-i]).T.dot(layer_delta[-i])
				self.weights[i]+= (lr/len(X))*dJdW[i]


		import matplotlib.pyplot as plt
		for i in range(len(cost)):
			plt.plot(i,cost[i],'k+')
		plt.show()

		#print("Final weights", self.weights[0])
		"""
        #Forward
		for i in self.weights:
			forward.append(nonlin(forward[-1]*i))


		error=forward[-1]-Y

		#layer_error will propogate backwards, itll keep on changin to the value of error at given layer in the loop
		layer_error=error
		#Will save all changes of self.weights to implement it simultaneously at the end
		layer_delta=0

		#Backpropogation
		for i in range(1,len(self.layers)):

			if(i==1):
				layer_delta=layer_error*nonlin(forward[-i],True)
				layer_error=layer_delta.dot(self.weights[-i])
				continue
			#For appending from front side
			layer_delta=list([layer_error*nonlin(forward[-i],True),layer_delta])
			#New error
			layer_error=layer_delta[0].dot(self.weights[-i])


		for i in range(1,len(self.layers)):
			self.weights[-i]+=forward[-i-1].T.dot(layer_delta[-i])
        """

	def feed_forward(self,X):
			f=X
			forward={'a':[],'z':[]}
			forward['z'].append(f)
			forward['a'].append(f)
			"""for i in range(len(self.weights)):
														forward.append([])"""
			for i in range(len(self.weights)):
				#print(np.dot(f,self.weights[i]))
				#print(self.nonlin(np.dot(f,self.weights[i])))
				forward['z'].append(np.dot(f,self.weights[i]))
				f=self.nonlin(np.dot(f,self.weights[i]))
				forward['a'].append(f)
				#print(f)
			#print(np.array(forward[-1]).shape)
			#print(forward[-1])
			#print(len(forward[0]))
			#print(len(forward[0][-1]))

			return forward

	def predict(self,x):
		f=x
		for i in range(len(self.weights)):
			f=self.nonlin(np.dot(f,self.weights[i]))
		return f


if __name__ == '__main__':
	import numpy as np
	from keras.datasets import mnist
	from keras.utils import np_utils
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn.metrics import precision_score
	from sklearn.metrics import coverage_error
	(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

	pixels=X_train.shape[1]*X_train.shape[2]
	model=NeuralNetwork()
	#model.plot(X_train[:10])
	np.random.seed(0)
	model.add(784,input_dim=pixels)
	model.add(150)
	model.add(20)
	model.add(10)

	X_train=X_train.reshape(X_train.shape[0],pixels).astype('float32')
	X_test=X_test.reshape(X_test.shape[0],pixels).astype('float32')
	num_classes=np.unique(Y_train).shape[0]
	#Normalize Data
	X_train=X_train/255
	X_test=X_test/255

	#One hot encode-Each output type has its own node
	#print(Y_train[0])
	Y_train=np_utils.to_categorical(Y_train)
	#print(Y_train[0])
	Y_test=np_utils.to_categorical(Y_test)

	print(X_train.shape)

	#print("X:",X_train[0],"Y:",Y_train[0])
	model.fit(X_train,Y_train,lr=0.08,epochs=20,lamda=5)
	predictions=model.predict(X_test)
	print("Accuracy Score: ",coverage_error(Y_test,predictions))
	prediction=model.predict(X_train[0])
	print(predictions[0])
	print(np.argmax(prediction),np.argmax(Y_train[0]))
	im=model.weights[0]
	model.plot(im[0:10])



