from keras.engine.topology import Layer
import keras.backend as K

if K.backend() == 'tensorflow':
	import tensorflow as tf
	
class AbsoluteMaxPooling(Layer):
	def __init__(self, pool_size, **kwargs):
		self.pool_size = pool_size
		
		super(AbsoluteMaxPooling, self).__init___(**kwargs)
	
	def build(self, input_shape):
		self.input_channels = input_shape[3]
	
	def compute_output_shape(self, input_shape):
		return None, input_shape[1] // self.pool_size, input_shape[2] // self.pool_size, input_shape[3]
	
	def call(self, x, mask=None):
		x1 = K.pool2d(x, self.pool_size, strides=(self.pool_size, self.pool_size), border_mode='same', dim_ordering='tf', pool_mode='max')
		x2 = K.pool2d(-x, self.pool_size, strides=(self.pool_size, self.pool_size), border_mode='same', dim_ordering='tf', pool_mode='max')
		
		m1 = K.cast(K.greater(x1, x2), dtype='float32')
		m2 = K.cast(K.greater_equal(x2, x1), dtype='float32')
		
		out = x1 * m1 - x2 * m2
		