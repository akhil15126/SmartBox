from Mitigation.Mitigation import Mitigation
import numpy as np
import tensorflow as tf
import cv2


class GaussianBlur(Mitigation):
	def __init__(self, sess, model, height, width, channels=1, minval=0.0, maxval=1.0,save=None):
		super().__init__(sess,model,height,width,channels,minval,maxval,save)

		

	def mitigate(self,X):
		final=[]
		for a in X:
			b=cv2.GaussianBlur(a, (3,3), 0)
			final.append(b)
		final=np.array(final)
		self.mitigated=np.reshape(final, (-1,self.height,self.width,self.channels))
		return self.mitigated


