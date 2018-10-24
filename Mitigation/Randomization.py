from Mitigation.Mitigation import Mitigation
import numpy as np
import tensorflow as tf
import random
import cv2


class Randomization(Mitigation):
	def __init__(self, sess, model, height, width, channels=1, minval=0.0, maxval=1.0,save=None):
		super().__init__(sess,model,height,width,channels,minval,maxval,save)



	def mitigate(self,X):
		final=[]
		X=((X-self.minval)/(self.maxval-self.minval))*255
		for a in X:
			a=a.reshape((self.height, self.width, self.channels))
			range_list_w=random.sample(range(self.width-4, self.width),3)
			range_list_h=random.sample(range(self.height-4, self.height),3)
			resize_image_list=[]
			for i in range_list_w:
				for j in range_list_h:
					img=cv2.resize(a, (2*i,2*j), interpolation=cv2.INTER_CUBIC)
					img=cv2.resize(img, (i,j), interpolation=cv2.INTER_AREA)
					resize_image_list.append(img)
			pad_image_list=[]
			for img in resize_image_list:
				w_=self.width-img.shape[1]
				h_=self.height-img.shape[0]
				wrange=range(0,w_+1)
				hrange=range(0,h_+1)
				for w in wrange:
					for h in hrange:
						if(self.channels==1):
							img2=np.pad(img, ((h, h_-h), (w, w_-w)), 'constant', constant_values=0)
							pad_image_list.append(img2)
						else:
							img2=np.pad(img, ((h, h_-h), (w, w_-w), (0,0)), 'constant', constant_values=0)
							pad_image_list.append(img2)
			img=pad_image_list[random.randint(0, len(pad_image_list)-1)]
			image=np.reshape(img,(self.height, self.width, self.channels)).astype(np.uint8)
			final.append(image)

		self.mitigated=np.array(final)
		self.mitigated=(self.mitigated/255)*(self.maxval-self.minval)+self.minval
		return self.mitigated

	

