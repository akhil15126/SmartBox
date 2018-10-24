import numpy as np
import tensorflow as tf
import random
import keras
from sklearn.svm import LinearSVC
from Detection.Detection import Detection
from CustomExceptions import *



class PCADetect(Detection):

	def __init__(self, sess, model, height, width, attack, trainX=None, trainY=None, channels=1, minval=0.0, maxval=1.0, restore=None,save=None):
		super().__init__(sess,model,height,width,channels,minval,maxval)

		if (trainX is None and restore is None):
			raise TrainingDataNotProvidedException
		
		self.attack=attack
		
		self.trainX=trainX
		self.trainY=trainY
		self.restore=restore
		self.save=save
		self.project_matrix=0
		self.mu=0
		self.classifier=LinearSVC()


	def train_PCA(self):
		m,w,h,k=self.trainX.shape
		data=np.reshape(self.trainX, (m, w*h*k)).T
		matrix=tf.placeholder(tf.float32, shape=data.shape)
		mean_var=tf.reduce_mean(matrix, 1)
		cov=tf.subtract(matrix,tf.reshape(mean_var, (w*h*k, 1)))
		cov_var=tf.divide(tf.matmul(cov, tf.transpose(cov)), tf.constant(m-1, dtype=tf.float32))
		e,v=tf.self_adjoint_eig(cov_var)
		mean, eig_values, eig_vectors=self.sess.run((mean_var, e, v), feed_dict={matrix:data})
		eig_values=np.maximum(eig_values, 0)
		a=np.argsort(eig_values).tolist()
		a.reverse()
		self.project_matrix=eig_vectors[:,a]
		self.mu=np.reshape(mean, (w*h*k, 1))



	def get_features(self, data):
		m,w,h,k=data.shape
		data=np.reshape(data, (m, w*h*k)).T
		data-=self.mu
		return np.matmul(self.project_matrix.T, data).T
		


	def detect(self,X):
		adv_data=self.attack.attack(self.trainX,self.trainY)
		self.train_PCA()
		x,y=self.prepare_data(self.trainX,adv_data)
		x=self.get_features(x)
		self.classifier.fit(x,np.argmax(y,axis=1))

		features=self.get_features(X)
		prediction=self.classifier.predict(features)
		# print(prediction.shape)
		self.detected=[]
		for a in prediction:
			if(a==0):
				self.detected.append(np.eye(2)[0])
			else:
				self.detected.append(np.eye(2)[1])
		self.detected=np.array(self.detected)

			
		

		return self.detected















