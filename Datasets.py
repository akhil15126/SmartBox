import numpy as np
import os
import cv2
import pickle
from CustomExceptions import DatasetNotLoadedException

class Datasets:
	def __init__(self,height,width,channels,restore=None):
		self.restore=restore
		self.height=height
		self.width=width
		self.channels=channels
		self.loaded=0
		if(restore!=None):
			dataset=pickle.load(open(restore, 'rb'))
			self.location=dataset.location
			self.images_train=dataset.images_train
			self.genuine_imposter=dataset.genuine_imposter
			self.images_test=dataset.images_test
			self.targets=dataset.targets
			self.images_valid=dataset.images_valid
			self.labels_train=dataset.labels_train
			self.labels_test=dataset.labels_test
			self.labels_valid=dataset.labels_valid
			self.maxval=dataset.maxval
			self.minval=dataset.minval
			self.train_split=dataset.train_split
			self.valid_split=dataset.valid_split
			self.test_split=dataset.test_split
			self.classes=dataset.classes
			self.loaded=1
		
	def get_info(self):
		if(not self.loaded):
			raise DatasetNotLoadedException

		print("\n")
		print("------------------------ LOADED DATASET INFORMATION ------------------------")
		print("\n")
		print("Dataset Location : "+str(self.location))
		print("Max value of pixels : "+str(self.maxval))
		print("Min value of pixels : "+str(self.minval))
		print("Training split : "+str(self.train_split*100)+"%")
		print("Validation split : "+str(self.valid_split*100)+"%")
		print("Testing split : "+str(self.test_split*100)+"%")
		print("Number of images used for training : "+str(len(self.images_train)))
		print("Number of images used for validation : "+str(len(self.images_valid)))
		print("Number of images used for testing : "+str(len(self.images_test)))
		print("Height : "+str(self.height))
		print("Width : "+str(self.width))
		print("Channels : "+str(self.channels))
		print("\n")
		print("---------------------------------------------------------------------------")
		print("\n")




			

	def load_dataset(self,location,train_split=0.8,valid_split=0.1,test_split=0.1,minval=0.0,maxval=1.0,save=None):
		folders=os.listdir(location)
		count=1
		self.location=location
		self.images_train=[]
		self.genuine_imposter=[]
		self.images_test=[]
		self.targets=[]
		self.images_valid=[]
		self.labels_train=[]
		self.labels_test=[]
		self.labels_valid=[]
		self.maxval=maxval
		self.minval=minval
		# classes=os.listdir(location)
		folders=[fol for fol in folders if fol[0]!='.']
		self.train_split=train_split
		self.valid_split=valid_split
		self.test_split=test_split
		self.classes=len(folders)


		for fol in folders:
			label=count
			files=os.listdir(location+fol)
			proper_files=np.array(files)
			np.random.shuffle(proper_files)
			data=[]
			for f in proper_files:
				try:
					img=cv2.imread(location+fol+"/"+f)
					if(len(img.shape)!=3):
						continue
				except:
					# print("YES")
					continue
				if(self.channels==1):
					img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img=cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
				data.append(img)
			l=len(data)
			indeces=np.arange(l)
			np.random.shuffle(indeces)
			data=np.array(data)
			train=int(self.train_split*l)
			valid=int(self.valid_split*l)
			train_indeces=indeces[:train]
			valid_indeces=indeces[train:(train+valid)]
			test_indeces=indeces[(train+valid):]
			self.images_train+=list(data[train_indeces])
			self.images_valid+=list(data[valid_indeces])
			self.images_test+=list(data[test_indeces])
			self.labels_train+=[np.eye(self.classes)[label-1]]*len(data[train_indeces])
			self.labels_valid+=[np.eye(self.classes)[label-1]]*len(data[valid_indeces])
			self.labels_test+=[np.eye(self.classes)[label-1]]*len(data[test_indeces])
			count+=1

		self.images_train=np.array(self.images_train)
		self.images_test=np.array(self.images_test)
		self.images_valid=np.array(self.images_valid)
		self.labels_train=np.array(self.labels_train)
		self.labels_test=np.array(self.labels_test)
		self.labels_valid=np.array(self.labels_valid)

		self.images_train=(self.images_train.astype(np.float)/255.0)*(maxval-minval)+minval
		self.images_test=(self.images_test.astype(np.float)/255.0)*(maxval-minval)+minval
		self.images_valid=(self.images_valid.astype(np.float)/255.0)*(maxval-minval)+minval
				

		indeces=np.arange(self.images_train.shape[0])
		np.random.shuffle(indeces)
		self.images_train=self.images_train[indeces]
		self.labels_train=self.labels_train[indeces]

		indeces=np.arange(self.images_valid.shape[0])
		np.random.shuffle(indeces)
		self.images_valid=self.images_valid[indeces]
		self.labels_valid=self.labels_valid[indeces]

		indeces=np.arange(self.images_test.shape[0])
		np.random.shuffle(indeces)
		self.images_test=self.images_test[indeces]
		self.labels_test=self.labels_test[indeces]

		self.images_train=self.images_train.reshape((-1, self.height, self.width, self.channels))
		self.images_valid=self.images_valid.reshape((-1, self.height, self.width, self.channels))
		self.images_test=self.images_test.reshape((-1, self.height, self.width, self.channels))

		indeces=np.arange(self.images_test.shape[0])
		np.random.shuffle(indeces)
		l=self.images_test.shape[0]
		lh=int(l/2)
		for i in range(l):
			m=np.argmax(self.labels_test[i])
			newm=m
			while(newm==m):
				newm=np.random.randint(self.classes)
			self.targets.append(np.eye(self.classes)[newm])
			if(i<lh):
				self.genuine_imposter.append(0)
			else:
				self.genuine_imposter.append(1)


		self.targets=np.array(self.targets)
		self.genuine_imposter=np.array(self.genuine_imposter)
		indeces=np.arange(self.images_test.shape[0])
		np.random.shuffle(indeces)
		self.images_test=self.images_test[indeces]
		self.labels_test=self.labels_test[indeces]
		self.genuine_imposter=self.genuine_imposter[indeces]
		self.targets=self.targets[indeces]
		self.loaded=1
		if(save!=None):
			temp_pkl = open(save,'wb')
			pickle.dump(self,temp_pkl)
			temp_pkl.close()
		return (self.images_train,self.labels_train,self.images_valid,self.labels_valid,self.images_test,self.labels_test,self.targets,self.genuine_imposter)


