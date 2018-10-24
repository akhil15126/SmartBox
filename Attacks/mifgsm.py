import tensorflow as tf
import numpy as np
import cv2
from Attacks.Attack import Attack
### LABEL LEAKING IMPLEMENT PREVENTION


class MIFGSM(Attack):
	def __init__(self, sess, model, eps=0.3,decay_factor=1,iterations=10,step=0.06,save=None,minval=0,maxval=1):
		super().__init__(sess,model,save,minval,maxval)
		self.eps=eps
		self.decay_factor=decay_factor
		self.step=step
		self.iterations=iterations


	def setup(self,x,y):
		momentum=0
		adv=x
		for i in range(self.iterations):

			preds=self.model.get_logits(adv)
			loss=tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=y)
			grad=tf.gradients(loss,adv)
			grad=grad[0]
			divisor=tf.maximum(tf.cast(1e-12, grad.dtype), tf.reduce_mean(tf.abs(grad), list(range(1,len(grad.shape))), keep_dims=True))
			grad=grad/divisor
			momentum=self.decay_factor*momentum + grad
			grad=tf.sign(momentum)
			grad=self.step*grad
			adv=adv+grad
			adv=x+tf.clip_by_value((adv-x), -1*self.eps, self.eps)
			adv=tf.clip_by_value(adv, self.minval, self.maxval)
			adv=tf.stop_gradient(adv)
		
		return adv,adv-x

	def attack(self, X_test, Y_test):
		x=tf.placeholder('float', shape=X_test.shape)
		y=tf.placeholder('float')
		a=self.setup(x,y)
		self.adversaries,self.perturbation=self.sess.run(a, feed_dict={x:X_test, y:Y_test})
		return self.adversaries




