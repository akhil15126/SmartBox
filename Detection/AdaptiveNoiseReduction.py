from Mitigation.Mitigation import Mitigation
from Detection.Detection import Detection
import numpy as np
import cv2
import tensorflow as tf
import math

class AdaptiveNoiseReduction(Detection, Mitigation):
	def __init__(self, sess, model, height, width, channels, minval=0.0, maxval=1.0, save=None):
		Detection.__init__(self, sess, model, height, width, channels, minval, maxval)
		Mitigation.__init__(self, sess, model, height, width, channels, minval, maxval, save)

		
	def pairing(self, m, n):
		return int(m*256+n)


	def scalarQuantization(self, x, step):
		x=np.floor(x/step)
		x=x*step
		x=(x/255)*(self.maxval-self.minval)+self.minval
		return x

	def calculate_entropy(self, x):
		x=np.moveaxis(x,-1,0)
		entropy=0
		# print(x.shape)
		for i in range(len(x)):
			filtered=cv2.blur(x[i],(5,5))
			x[i]=x[i].astype(int)
			filtered=filtered.astype(int)
			arr=np.zeros((256*256,))
			for a,b in zip(x[i],filtered):
				for m,n in zip(a,b):
					arr[self.pairing(m,n)]+=1
			p=np.array(arr)
			p=p/(self.height*self.width)
			p[p==0]=1
			entropy+=np.sum(p*np.log2(p))


		entropy=entropy/len(x)
		entropy=-1*entropy
		return entropy


	def apply_filter(self, x):
		kernel=np.zeros((5,5))
		kernel[2]=1
		kernel[:,2]=1
		return cv2.filter2D(x, -1, kernel)

	
	def CloserFilter(self,x,y,z):
		#print(x.shape,y.shape,z.shape)
#		x=x.reshape(z.shape)
		z=z.reshape(x.shape)
		fin=np.zeros(x.shape)
		a=np.abs(x-y)
		b=np.abs(x-z)
		fin[a<b]=y[a<b]
		fin[b<a]=z[b<a]
		return fin

	def helper(self, X):
		Y=[]
		newX=[]
		for x in X:
		# x=X
			currentlabel=np.argmax(self.model.model.predict(x.reshape((-1,self.height,self.width,self.channels))), 1)
			img_entropy=self.calculate_entropy(((x-self.minval)/(self.maxval-self.minval))*255)
			# print(img_entropy)
				# new_img=cv2.blur(x,(5,5))
			if(img_entropy<8.5):
				new_img=self.scalarQuantization(((x-self.minval)/(self.maxval-self.minval))*255,128)
			elif img_entropy<9.5:
				new_img=self.scalarQuantization(((x-self.minval)/(self.maxval-self.minval))*255,64)
			else:
				new_img=self.scalarQuantization(((x-self.minval)/(self.maxval-self.minval))*255,50)
				new_img1=self.apply_filter(new_img)
				new_img=self.CloserFilter(x, new_img, new_img1)
			newX.append(new_img)
			newlabel=np.argmax(self.model.model.predict(new_img.reshape((-1,self.height,self.width,self.channels))), 1)
			if(currentlabel==newlabel):
				Y.append(np.eye(2)[0])
			else:
				Y.append(np.eye(2)[1])

		self.detected=np.array(Y)
		self.mitigated=np.array(newX)
		return self.detected, self.mitigated

	def detect(self,X):
		a,_=self.helper(X)
		return a

	def mitigate(self,X):
		_,a=self.helper(X)
		return a



	def analyse(self,Y_test=None,X_test=None,save=""):
		if (X_test is None and Y_test is not None):
			return Detection.analyse(self,Y_test)
		else:
			return Mitigation.analyse(self,X_test, Y_test, save)

	def print_analysis_results(self,Y_test=None,X_test=None,save=""):
		if (X_test is None and Y_test is not None):
			return Detection.print_analysis_results(self,Y_test)
		else:
			return Mitigation.print_analysis_results(self,X_test, Y_test, save)




