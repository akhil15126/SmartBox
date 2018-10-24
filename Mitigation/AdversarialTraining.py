from Mitigation.Mitigation import Mitigation
from Model import GeneralModel
import numpy as np
import tensorflow as tf
import cv2


class AdversarialTraining(Mitigation):
	def __init__(self, sess, model, height, width, trainX=None, trainY=None,attack=None,channels=1, minval=0.0, maxval=1.0,save=None,epochs=100,batch_size=100,restore=None,modelsave=None):
		super().__init__(sess,model,height,width,channels,minval,maxval,save)
		if (trainX is None and restore is None):
			raise TrainingDataNotProvidedException
		self.trainX=trainX
		self.trainY=trainY
		self.attack=attack
		self.modelsave=modelsave
		self.restore=restore
		self.num_labels=model.num_labels
		self.epochs=epochs
		self.batch_size=batch_size

		


	def mitigate(self,X):
		adv_x=self.attack.attack(self.trainX,self.trainY)
		model=GeneralModel(self.height,self.width,self.num_labels,self.channels,restore=self.restore)
		if self.restore is None:
			trainX=np.concatenate((self.trainX,adv_x),axis=0)
			trainY=np.concatenate((self.trainY,self.trainY),axis=0)
			model.train((trainX,trainY,None,None),self.epochs, self.batch_size, modeldir=self.modelsave)

		self.mitigated=model
		return self.mitigated

		


