import numpy as np
import tensorflow as tf
import keras
from keras import backend as K


class GeneralModel():
	def __init__(self,height,width,classes,channels,restore=None):
		self.height=height
		self.width=width
		self.num_channels=channels
		self.num_labels=classes
		self.model=keras.models.Sequential()

		self.model.add(keras.layers.Conv2D(filters=32, kernel_size=5, input_shape=(self.height,self.width,self.num_channels), activation="relu"))
		self.model.add(keras.layers.MaxPooling2D(pool_size=2))
		self.model.add(keras.layers.Conv2D(filters=64, kernel_size=5, activation="relu"))
		self.model.add(keras.layers.MaxPooling2D(pool_size=2))
		self.model.add(keras.layers.Flatten())
		self.model.add(keras.layers.Dense(1024))
		self.model.add(keras.layers.Activation('relu'))
		self.model.add(keras.layers.Dropout(0.4))
		self.model.add(keras.layers.Dense(self.num_labels))
		self.layer_outputs={}
		for i, a in enumerate(self.model.layers):
			l=type(a).__name__
			if l not in self.layer_outputs:
				b=list()
				self.layer_outputs[l]=b
				self.layer_outputs[l].append(K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[i].output]))
			else:
				self.layer_outputs[l].append(K.function([self.model.layers[0].input, K.learning_phase()],[self.model.layers[i].output]))
		if(restore!=None):
			self.model.load_weights(restore)

	def train(self, data,epochs, batch_size, learning_rate=0.01,momentum=0.9,decay=1e-6,nesterov=True,modeldir=None):
		self.images_train,self.labels_train,self.images_valid,self.labels_valid=data
		self.epochs=epochs
		self.batch_size=batch_size

		def loss(actual,predicted,galleryx=None,galleryy=None):
			return tf.nn.softmax_cross_entropy_with_logits(logits=predicted, labels=actual)

		sgd = keras.optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)

		self.model.compile(loss=loss, optimizer=sgd, metrics=["accuracy"])
		if self.images_valid is None:
			self.model.fit(self.images_train, self.labels_train, batch_size=self.batch_size, nb_epoch=self.epochs, shuffle=True)
		else:
			self.model.fit(self.images_train, self.labels_train, batch_size=self.batch_size, validation_data=(self.images_valid,self.labels_valid), nb_epoch=self.epochs, shuffle=True)

		if(modeldir!=None):
			self.model.save(modeldir)

		return self.model

	def get_logits(self, x):
		return self.model(x)

	def get_layer_output(self, layer, x, phase):
		if layer not in self.layer_outputs:
			return None
		else:
			outputs=[]
			for func in self.layer_outputs[layer]:
				outputs.append(func([x, phase]))
		return outputs
