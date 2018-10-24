import tensorflow as tf
import numpy as np
import cv2
# from cnn import GeneralModel
from Attacks.Attack import Attack
# folder="/Users/akhilgoel/Desktop/smlassignment/mnist/"

# from tensorflow.examples.tutorials.mnist import input_data

# mnist=input_data.read_data_sets(folder, one_hot=True)
### LABEL LEAKING IMPLEMENT PREVENTION


class FGSM(Attack):
	def __init__(self, sess, model, eps=0.3,save=None,minval=0,maxval=1):
		super().__init__(sess,model,save,minval,maxval)
		self.eps=eps

	def setup(self,x,y):
		preds=self.model.get_logits(x)
		loss=tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
		grad=tf.gradients(loss,x)
		grad=grad[0]
		grad=tf.sign(grad)
		grad=self.eps*grad
		adv=x+grad
		adv=tf.clip_by_value(adv,self.minval,self.maxval)
		return adv,adv-x

	def attack(self, X_test, Y_test):
		x=tf.placeholder('float', shape=X_test.shape)
		y=tf.placeholder('float')
		a=self.setup(x,y)
		self.adversaries,self.perturbation=self.sess.run(a, feed_dict={x:X_test, y:Y_test})
		
		return self.adversaries




