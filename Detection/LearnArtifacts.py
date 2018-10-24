import numpy as np
import tensorflow as tf
import math
import keras
from Detection.Detection import Detection
from CustomExceptions import *

class BinaryModel:
	def __init__(self, shape, restore=None):
		self.shape=shape
		self.model=keras.models.Sequential()
		self.model.add(keras.layers.Dense(400, input_dim=self.shape, activation='relu'))
		self.model.add(keras.layers.Dense(150, activation='relu'))
		self.model.add(keras.layers.Dense(2))
		if(restore!=None):
			self.model.load_weights(restore)

	def train(self, X_train,Y_train,batch_size=100,epochs=100, save=None):

		def loss(actual, predicted):
			return tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=actual)
		self.model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
		self.model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, shuffle=True)
		if(save!=None):
			self.model.save(save)
		return self.model

class LearnArtifactsModel(Detection):

	def __init__(self, sess, model, height, width, attack, trainX=None, trainY=None, batch_size=100, epochs=100,  channels=1, minval=0.0, maxval=1.0, restore=None,save=None):
		super().__init__(sess,model,height,width,channels,minval,maxval)

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

	
	def detect(self,X):
		l=self.trainX.shape[0]
		adv_tr=self.attack.attack(self.trainX,self.trainY)
		
		if self.restore is None:
			for a,b in zip(self.trainX,adv_tr):
				feature_nor=self.model.get_layer_output('Dense', a.reshape(-1,self.height,self.width,self.channels), 0)
				feature_adv=self.model.get_layer_output('Dense', b.reshape(-1,self.height, self.width, self.channels), 0)   
				fn=feature_nor[-2]
				fa=feature_adv[-2]
				fn=fn[0].reshape((fn[0].shape[1]))
				fa=fa[0].reshape((fa[0].shape[1]))
				self.X_train.append(fn)
				self.Y_train.append(np.eye(2)[0])
				self.X_train.append(fa)
				self.Y_train.append(np.eye(2)[1])
			self.X_train=np.array(self.X_train)
			self.Y_train=np.array(self.Y_train)
			print(self.X_train.shape)
			print(self.Y_train.shape)



		for a in X:
			feature_test=self.model.get_layer_output('Dense', a.reshape(-1,self.height,self.width,self.channels), 0)
			ft=feature_test[-2]
			ft=ft[0].reshape((ft[0].shape[1]))
			self.X_test.append(ft)
		self.X_test=np.array(self.X_test)
		print(self.X_test.shape)
		model=BinaryModel(self.X_train.shape[1],restore=self.restore)

		if self.restore is None:
			model.train(self.X_train,self.Y_train,self.batch_size, self.epochs, save=self.save)
		pred=model.model.predict(self.X_test)
		self.detected=np.copy(pred)

		return self.detected








