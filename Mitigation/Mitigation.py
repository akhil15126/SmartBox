from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from CustomExceptions import DetectNotRunProperlyException

class Mitigation(metaclass=ABCMeta):
	def __init__(self, sess, model, height, width, channels, minval=0.0, maxval=1.0, save=None):
		self.sess=sess
		self.model=model
		self.minval=minval
		self.maxval=maxval
		self.height=height
		self.width=width
		self.channels=channels
		self.mitigated=None
		self.save=save
	

	@abstractmethod
	def mitigate():
		pass

	def return_mitigated(self):
		if self.mitigated is None:
			raise MitigateNotRunProperlyException
		return self.mitigated


	def analyse(self, X_test=None, Y_test=None, save=""):
		if save=="":
			save=self.save

		if(self.mitigated is None):
			raise MitigateNotRunProperlyException
		if self.__class__.__name__=="AdversarialTraining":
			mitigated=self.mitigated.model.predict(X_test)
			mitigated=np.argmax(mitigated, axis=1)
		else:
			mitigated=self.model.model.predict(self.mitigated)
			mitigated=np.argmax(mitigated, axis=1)
		original=self.model.model.predict(X_test)
		original=np.argmax(original, axis=1)
		
		actual=np.argmax(Y_test,axis=1)
		correct_orig=np.equal(original,actual)
		accuracy_orig=np.mean(correct_orig)
		correct_mit=np.equal(mitigated,actual)
		accuracy_mit=np.mean(correct_mit)
		a=dict()
		a["beforemit"]=accuracy_orig
		a["aftermit"]=accuracy_mit
		if save is not None and not self.__class__.__name__=="AdversarialTraining":
			from GeneralUtils import save_images_for_comparison
			save_images_for_comparison(self.__class__.__name__, actual, (X_test-self.minval)/(self.maxval-self.minval), original, (self.mitigated-self.minval)/(self.maxval-self.minval), mitigated, save)
		return a

	def print_analysis_results(self, X_test, Y_test, save=""):
		a=self.analyse(X_test=X_test,Y_test=Y_test,save=save)
		print("\n")
		print("------------------------ MITIGATION PERFORMANCE REPORT ------------------------")
		print("\n")
		print("Accuracy before mitigation : ", a["beforemit"])
		print("Accuracy after mitigation : ", a["aftermit"])
		print("\n")
		print("-------------------------------------------------------------------------------")
		print("\n")
		return a



	
