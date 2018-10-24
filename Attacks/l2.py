import tensorflow as tf
import numpy as np
from Attacks.Attack import Attack

import cv2



class L2(Attack):
	def __init__(self, sess,model,save=None,batch_size=4,minval=0.0,maxval=1.0):
		super().__init__(sess,model,save,minval,maxval)
		self.sess=sess
		self.model=model
		self.batch_size=batch_size
		self.leftovers=0
		self.const=tf.placeholder(tf.as_dtype('float32'), shape=[self.batch_size])


	def setup(self):
		self.pert=tf.Variable(np.zeros(shape=self.img.shape, dtype=np.dtype('float32')))
		tmp=self.pert+self.img

		self.adv=(tf.tanh(tmp)+1)/2
		self.orig=(tf.tanh(self.img)+1)/2

		self.adv=self.adv*(self.maxval-self.minval)+self.minval
		self.orig=self.orig*(self.maxval-self.minval)+self.minval

		self.output=self.model.get_logits(self.adv)

		self.l2dist=tf.reduce_sum(tf.square(self.adv-self.orig), list(range(1,len(self.img.shape))))

		target_score=tf.reduce_sum((self.target)*self.output,1)
		max_score=tf.reduce_max((tf.cast(tf.logical_xor(tf.cast(self.target,tf.as_dtype('bool')),True), tf.as_dtype('float32')))*self.output-self.target*10000, 1)

		loss1=tf.nn.relu(max_score-target_score)
		loss1=self.const*loss1

		self.l2distance=tf.reduce_sum(self.l2dist)
		self.loss1=tf.reduce_sum(loss1)
		self.loss=self.loss1+self.l2distance

		prevvars=list()
		for x in tf.global_variables():
			prevvars.append(x.name)
		# prevvars=set(x.name for x in tf.global_variables())
		optimizer=tf.train.AdamOptimizer(7e-3)
		self.train=optimizer.minimize(self.loss, var_list=[self.pert])
		# currentvars=tf.global_variables()

		new_vars=list()
		new_vars.append(self.pert)
		for x in tf.global_variables():
			if x.name in prevvars:
				continue
			else:
				new_vars.append(x)
		# new_vars=[x for x in currentvars if x.name not in prevvars]
		self.init=tf.variables_initializer(var_list=new_vars)

	def attack(self,X,targets):
		# if(X.shape[0]%self.batch_size!=0):
		self.leftovers=X.shape[0]%self.batch_size
			
		self.labels=targets.shape[-1]
		self.img=tf.placeholder(tf.as_dtype('float32'), shape=(self.batch_size,)+X.shape[1:])
		self.target=tf.placeholder(tf.as_dtype('float32'), shape=(self.batch_size,self.labels))
		self.setup()

		
		X=X-self.minval
		X=X/(self.maxval-self.minval)
		iterator=0
		X=2*X-1
		X=np.arctanh(X*0.999999)
		if self.leftovers!=0:
			X=np.concatenate((X,X[:(self.batch_size-self.leftovers)]))
			targets=np.concatenate((targets,targets[:(self.batch_size-self.leftovers)]))

		self.adversaries=np.empty(X.shape, X.dtype)
		self.perturbation=np.empty(X.shape, X.dtype)

		
		while(iterator<X.shape[0]):

			lower_bound=np.zeros(self.batch_size)
			const=np.ones(self.batch_size)*1e-2
			upper_bound=np.ones(self.batch_size)*1e10

			o_bestl2=[1e10]*self.batch_size
			o_bestscorelabel=[-1]*self.batch_size
			o_bestadv=np.copy((X[iterator:iterator+self.batch_size])*(self.maxval-self.minval)+self.minval)


			for j in range(5):
				self.sess.run(self.init)
				bestl2=[1e10]*self.batch_size
				bestscorelabel=[-1]*self.batch_size
				for i in range(1000):
					tr,l,l2s,scores,advimg=self.sess.run([self.train, self.loss, self.l2dist, self.output,self.adv], feed_dict={self.img:X[iterator:iterator+self.batch_size],self.target:targets[iterator:iterator+self.batch_size],self.const:const})
					for en,(l2,sc,ii,t) in enumerate(zip(l2s,scores,advimg,targets[iterator:iterator+self.batch_size])):
						tar=np.argmax(t)
						mysc=np.argmax(sc)
						if l2<bestl2[en] and mysc==tar:
							bestl2[en]=l2
							bestscorelabel[en]=mysc

							if l2<o_bestl2[en] and mysc==tar:
								o_bestl2[en]=l2
								o_bestscorelabel[en]=mysc
								o_bestadv[en]=ii

				# print("\n")
				for i in range(self.batch_size):
					if bestscorelabel[i]==np.argmax(targets[iterator:iterator+self.batch_size][i]):
						upper_bound[i]=min(upper_bound[i],const[i])
						const[i]=(lower_bound[i]+upper_bound[i])/2
					else:
						lower_bound[i]=max(lower_bound[i],const[i])
						if upper_bound[i]<1e9:
							const[i]=(lower_bound[i]+upper_bound[i])/2
						else:
							const[i]*=10
				o_bestl2=np.array(o_bestl2)
			o_bestl2=np.array(o_bestl2)
			self.adversaries[iterator:iterator+self.batch_size]=o_bestadv
			self.perturbation[iterator:iterator+self.batch_size]=self.sess.run(self.pert)
			iterator+=self.batch_size
			if(self.leftovers==0):
				print(str(iterator)+"/"+str(X.shape[0])+" done")
			else:
				print(str(min(iterator,X.shape[0]-self.batch_size+self.leftovers))+"/"+str(X.shape[0]-self.batch_size+self.leftovers)+" done")
		if(self.leftovers!=0):
			self.adversaries=self.adversaries[:X.shape[0]-self.batch_size+self.leftovers]
			self.perturbation=self.perturbation[:X.shape[0]-self.batch_size+self.leftovers]
		print(self.adversaries.shape)
		return self.adversaries



		




