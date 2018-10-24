import tensorflow as tf
import numpy as np
import cv2
# from cnn import GeneralModel
from Attacks.Attack import Attack
# folder="/Users/akhilgoel/Desktop/smlassignment/mnist/"

# from tensorflow.examples.tutorials.mnist import input_data

# mnist=input_data.read_data_sets(folder, one_hot=True)

### LABEL LEAKING IMPLEMENT PREVENTION



class PGD(Attack):
	def __init__(self, sess, model, eps=0.3,step=0.01,iterations=40,save=None,minval=0,maxval=1):
		super().__init__(sess,model,save,minval,maxval)
		self.eps=eps
		self.step=step
		self.iterations=iterations

	def setup(self,x,y):
		pert=tf.random_uniform(tf.shape(x), -1*self.eps, self.eps)
		for i in range(0,self.iterations):
			adv=x+pert
			preds=self.model.get_logits(adv)
			loss=tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
			grad,=tf.gradients(loss,adv)
			# grad=grad[0]
			grad=tf.sign(grad)
			grad=self.step*grad
			adv+=grad
			adv=tf.clip_by_value(adv,self.minval,self.maxval)
			pert+=grad
			pert=tf.clip_by_value(pert,-1*self.eps,self.eps)
		adv_x=x+pert
		adv_x=tf.clip_by_value(adv_x,self.minval,self.maxval)
		return adv_x,adv_x-x

	def attack(self, X_test, Y_test):
		x=tf.placeholder('float', shape=X_test.shape)
		y=tf.placeholder('float')
		a=self.setup(x,y)
		self.adversaries,self.perturbation=self.sess.run(a, feed_dict={x:X_test, y:Y_test})
		return self.adversaries





