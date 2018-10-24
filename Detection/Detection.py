from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from CustomExceptions import DetectNotRunProperlyException

class Detection(metaclass=ABCMeta):
	def __init__(self, sess, model, height, width, channels, minval=0.0, maxval=1.0):
		self.sess=sess
		self.model=model
		self.minval=minval
		self.maxval=maxval
		self.height=height
		self.width=width
		self.channels=channels
		self.detected=None
		self.TP=0
		self.FP=0
		self.TN=0
		self.FN=0
		self.Y_true=None
	

	@abstractmethod
	def detect():
		pass

	def return_detected(self):
		if self.detected is None:
			raise DetectNotRunProperlyException
		return self.detected

	def prepare_data(self,X,adv):
		X_detect=[]
		Y_detect=[]
		for a in X:
			X_detect.append(a)
			Y_detect.append(np.eye(2)[0])
		for a in adv:
			X_detect.append(a)
			Y_detect.append(np.eye(2)[1])

		X_detect=np.array(X_detect)
		Y_detect=np.array(Y_detect)

		indeces=np.arange(X_detect.shape[0])
		np.random.shuffle(indeces)
		X_detect=X_detect[indeces]
		Y_detect=Y_detect[indeces]
		
		return X_detect,Y_detect
		

	def analyse(self,Y_test=None):
		if Y_test is None:
			Y_test=self.Y_true
		else:
			self.Y_true=Y_test
		if self.detected is None:
			raise DetectNotRunProperlyException
		if Y_test is None:
			raise TrueLabelsNotProvidedException

		if (self.TP==self.FP and self.FP==0):
			pred=np.argmax(self.detected, axis=1)
			actual=np.argmax(Y_test,axis=1)
			for a,b in zip(actual,pred):
				if(a==b and a==1):
					self.TP+=1
				if(a!=b and a==1):
					self.FN+=1
				if(a==b and a==0):
					self.TN+=1
				if(a!=b and a==0):
					self.FP+=1


		a=dict()
		a["TP"]=self.TP
		a["FP"]=self.FP
		a["TN"]=self.TN
		a["FN"]=self.FN
		a["Precision"]=self.TP/(self.TP+self.FP)
		a["Recall"]=self.TP/(self.TP+self.FN)
		a["Detection_acc"]=(self.TP+self.TN)/(self.TP+self.FP+self.TN+self.FN)
		return a

	def print_analysis_results(self,Y_test=None):
		a=self.analyse(Y_test)
		print("\n")
		print("------------------------ DETECTION PERFORMANCE REPORT ------------------------")
		print("\n")
		print("True Positives : ", a["TP"])
		print("False Positives : ", a["FP"])
		print("True Negatives : ", a["TN"])
		print("False Negatives : ", a["FN"])
		print("Precision : ", a["Precision"])
		print("Recall : ", a["Recall"])
		print("Detection Accuracy : ", a["Detection_acc"])
		print("\n")
		print("------------------------------------------------------------------------------")
		print("\n")
		return a


	