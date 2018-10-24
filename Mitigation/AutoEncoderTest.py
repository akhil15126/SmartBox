import numpy as np
import tensorflow as tf
import math
import keras
from Mitigation.Mitigation import Mitigation
from keras import backend as K
from keras import regularizers
from CustomExceptions import *

class BinaryModel:
	def __init__(self, shape, minval=0.0, maxval=1.0,restore=None):
		self.shape=shape
		self.minval=minval
		self.maxval=maxval
		self.model=keras.models.Sequential()
		self.model.add(keras.layers.Dense(800, input_dim=self.shape, activation='relu'))
		self.model.add(keras.layers.Dense(self.shape, activation=self.custom_activation))
		if(restore!=None):
			self.model.load_weights(restore)

	def custom_activation(self,x):
		return K.sigmoid(x)*(self.maxval-self.minval)+self.minval

	def train(self, X_train,Y_train,batch_size=100,epochs=100, save=None):
		self.model.compile(loss='mean_squared_error', optimizer='adagrad')
		self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, shuffle=True)
		return self.model


		if(save!=None):
			self.model.save(save)
		return self.model

class AutoEncoder(Mitigation):

	def __init__(self, sess, model, height, width, attack, trainX=None, trainY=None, batch_size=100, epochs=100,  channels=1, minval=0.0, maxval=1.0, restore=None,save=None,modelsave=None):
		super().__init__(sess,model,height,width,channels,minval,maxval,save)

		if (trainX is None and restore is None):
			raise TrainingDataNotProvidedException
		self.attack=attack
		self.X_train=[]
		self.Y_train=[]
		self.X_test=[]
		self.Y_test=[]
		self.trainX=trainX
		self.trainY=trainY
		self.restore=restore
		self.save=save
		self.batch_size=batch_size
		self.epochs=epochs
		self.modelsave=modelsave

	
	def mitigate(self,X):
		l=self.trainX.shape[0]
		adv_tr=self.attack.attack(self.trainX,self.trainY)
		if self.restore is None:
			self.X_train=np.reshape(adv_tr,(-1, self.height*self.width*self.channels))
			self.Y_train=np.reshape(self.trainX, (-1, self.height*self.width*self.channels))
		self.X_test=np.reshape(X, (-1,self.height*self.width*self.channels))
		print(self.X_test.shape)
		model=BinaryModel(self.X_train.shape[1],self.minval,self.maxval,restore=self.restore)

		if self.restore is None:
			model.train(self.X_train,self.Y_train,self.batch_size, self.epochs, save=self.modelsave)
		pred=model.model.predict(self.X_test)
		self.mitigated=np.copy(pred)
		self.mitigated=np.reshape(self.mitigated,(-1,self.height,self.width,self.channels))
	

		return self.mitigated











