from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
from CustomExceptions import AdversariesNotComputedProperlyException, PerturbationNotComputedProperlyException

class Attack(metaclass=ABCMeta):
	def __init__(self, sess, model, save=None, minval=0.0, maxval=1.0):
		self.perturbation=None
		self.sess=sess
		self.model=model
		self.save=save
		self.adversaries=None
		self.minval=minval
		self.maxval=maxval

	@abstractmethod
	def setup():
		pass

	@abstractmethod
	def attack():
		pass

	def analyse(self, X_test, Y_test, save=""):
		if save=="":
			save=self.save

		if(self.adversaries is None):
			raise AdversariesNotComputedProperlyException
		original=self.model.model.predict(X_test)
		original=np.argmax(original, axis=1)
		adversarial=self.model.model.predict(self.adversaries)
		adversarial=np.argmax(adversarial, axis=1)
		actual=np.argmax(Y_test,axis=1)
		correct_orig=np.equal(original,actual)
		accuracy_orig=np.mean(correct_orig)
		correct_adv=np.equal(adversarial,actual)
		accuracy_adv=np.mean(correct_adv)
		a=dict()
		a["ORIGINAL ACCURACY"]=accuracy_orig
		a["ADVERSARIAL ACCURACY"]=accuracy_adv
		if save is not None:
			from GeneralUtils import save_images_for_comparison
			save_images_for_comparison(self.__class__.__name__, actual, (X_test-self.minval)/(self.maxval-self.minval), original, (self.adversaries-self.minval)/(self.maxval-self.minval), adversarial, save)
		return a

	def print_analysis_results(self, X_test, Y_test, save=""):
		a=self.analyse(X_test,Y_test,save)
		print("\n")
		print("------------------------ ATTACK PERFORMANCE REPORT ------------------------")
		print("\n")
		print("Original Accuracy : ", a["ORIGINAL ACCURACY"])
		print("Adversarial Accuracy : ", a["ADVERSARIAL ACCURACY"])
		print("\n")
		print("---------------------------------------------------------------------------")
		print("\n")
		return a


	def get_perturbation(self):
		if(self.perturbation is None):
			raise PerturbationNotComputedProperlyException
		return self.perturbation

