import numpy as np
import tensorflow as tf
import random
import keras
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression as LR
from sklearn.calibration import CalibratedClassifierCV as ccv
from Detection.Detection import Detection
from CustomExceptions import *



class ConvFiltDetect(Detection):

	def __init__(self, sess, model, height, width, attack, trainX=None, trainY=None, maxpoollayers=3, channels=1, minval=0.0, maxval=1.0, restore=None,save=None):
		super().__init__(sess,model,height,width,channels,minval,maxval)

		if (trainX is None and restore is None):
			raise TrainingDataNotProvidedException
		self.attack=attack
		self.trainX=trainX
		self.trainY=trainY
		self.restore=restore
		self.save=save
		self.layers=[i for i in range(maxpoollayers)] 
		self.layers_svm=[] 
		self.stds=[] 
		self.project_matrixes=[] 
		self.mus=[] 



	def train_PCA(self):
		for layer in range(len(self.layers)):
			output=self.model.get_layer_output('MaxPooling2D', self.Npool, 0)[layer][0]
			m,w,h,k=output.shape
			data=np.reshape(output, (m, w*h*k)).T
			matrix=tf.placeholder(tf.float32, shape=data.shape)
			mean_var=tf.reduce_mean(matrix, 1)
			cov=tf.subtract(matrix,tf.reshape(mean_var, (w*h*k, 1)))
			cov_var=tf.divide(tf.matmul(cov, tf.transpose(cov)), tf.constant(m-1, dtype=tf.float32))
			e,v=tf.self_adjoint_eig(cov_var)
			mean, eig_values, eig_vectors=self.sess.run((mean_var, e, v), feed_dict={matrix:data})
			a=np.argsort(eig_values).tolist()
			a.reverse()
			W=eig_vectors[:,a]
			mean=np.reshape(mean, (w*h*k, 1))
			self.mus.append(mean)
			self.project_matrixes.append(W)
			new_result=tf.matmul(tf.transpose(tf.constant(W, dtype=tf.float32)), tf.subtract(matrix,tf.reshape(mean_var, (w*h*k, 1))))
			resultant=self.sess.run(new_result, feed_dict={matrix:data})
			std=np.std(resultant, axis=1)
			std=np.reshape(std, (w*h*k, 1))+1e-7
			self.stds.append(std)


	def extract_pca_coefficients(self, data, layer):
		data-=self.mus[layer]
		data=np.matmul(self.project_matrixes[layer].T, data)
		data/=self.stds[layer]
		return data.T


	def get_features(self, data, layer):
		n,w,h,k=data.shape
		data=np.reshape(data, (n, w*h*k))
		pca_coefficients=self.extract_pca_coefficients(np.copy(data.T), layer)
		data=np.reshape(data, (n, w*h, k))
		pca_coefficients=np.linalg.norm(np.reshape(pca_coefficients, (n,w*h,k)), ord=1, axis=1).astype(float)
		maximal_values=np.amax(data, axis=1)
		minimal_values=np.amin(data, axis=1)
		percentile_values=np.concatenate((np.percentile(data, 25, axis=1), np.percentile(data, 50, axis=1), np.percentile(data, 75, axis=1)), axis=1)
		return np.concatenate((pca_coefficients, maximal_values, minimal_values, percentile_values), axis=1)



	def train(self):
		current_layer=0
		self.train_PCA()
		while(current_layer<len(self.layers) and self.Npool.shape[0]>=self.Ptrain.shape[0]):
			mo,_,_,_=self.Npool.shape
			ma,_,_,_=self.Ptrain.shape
			pnormal=random.sample(range(0, mo),ma)
			train_data=np.concatenate((np.copy(self.Npool[pnormal,:,:,:]), self.Ptrain), axis=0)
			train_labels=np.zeros(len(pnormal)).tolist()+np.ones(ma).tolist()
			output=self.model.get_layer_output('MaxPooling2D', train_data, 0)[current_layer][0]
			train_data=self.get_features(output, current_layer)
			classifier=ccv(LinearSVC())
			classifier.fit(train_data, train_labels)
			scores=classifier.predict_proba(train_data[:len(pnormal)])[:,0]
			scores=scores[scores>0.5]
			threshold=np.median(scores)
			output=self.model.get_layer_output('MaxPooling2D', self.Npool, 0)[current_layer][0]
			npool_data=self.get_features(output, current_layer)
			a=classifier.predict(npool_data)
			scores=classifier.predict_proba(npool_data)[:,0]
			scores=(scores.astype(np.float32)/threshold).astype(int).tolist()
			self.layers_svm.append(classifier)
			indexes=[i for i in range(len(scores)) if scores[i]!=1]
			self.Npool=self.Npool[indexes,:,:,:]
			current_layer+=1





	def detect(self,X):
		l=self.trainX.shape[0]
		self.Npool=np.copy(self.trainX)
		num=int(self.Npool.shape[0]*0.2)
		pran=random.sample(range(0, self.Npool.shape[0]), num)
		self.Ptrain=self.attack.attack(self.Npool[pran],self.trainY[pran])
		self.train()
		adv_test=np.copy(X)
		current_layer=0
		m,_,_,_=adv_test.shape
		indexes=[i for i in range(m)]
		predict_labels=[0 for i in range(X.shape[0])]
		while(current_layer<len(self.layers_svm) and adv_test.shape[0]!=0):
			output=self.model.get_layer_output('MaxPooling2D', adv_test, 0)[current_layer][0]
			features=self.get_features(output, current_layer)
			classifier=self.layers_svm[current_layer]
			prediction=classifier.predict(features)
			index_pred=[i for i in range(len(prediction)) if prediction[i]==0]
			new_index=list(set([i for i in range(len(prediction))])-set(index_pred))
			adv_test=adv_test[new_index,:,:,:]
			indexes=[indexes[i] for i in new_index]
			current_layer+=1
		if(current_layer==len(self.layers_svm)):
			for i in indexes:
				predict_labels[i]=1
		self.detected=[]
		for i in predict_labels:
			if(i==0):
				self.detected.append(np.eye(2)[0])
			else:
				self.detected.append(np.eye(2)[1])

		self.detected=np.array(self.detected)

			
		

		return self.detected
















