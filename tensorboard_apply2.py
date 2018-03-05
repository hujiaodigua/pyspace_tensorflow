import tensorflow as tf
import numpy as np

 ## make up some data
 x_data= np.linspace(-1, 1, 300, dtype=np.float32)[:,np.newaxis]
 noise=  np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
 y_data= np.square(x_data) -0.5+ noise

def add_layer(
    inputs ,
    in_size,
    out_size,
    n_layer,    #这个参数用来标识层数
    activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name = 'layer%s'%n_layer  ## define a new var
    ## and so on ……

def add_layer(inputs ,
            in_size,
            out_size,n_layer,
            activation_function=None):
    ## add one more layer and return the output of this layer
    layer_name = 'layer%s'%n_layer
    with tf.name_scope('layer'):
         with tf.name_scope('weights'):
              Weights= tf.Variable(tf.random_normal([in_size, out_size]),name='W')
              # tf.histogram_summary(layer_name+'/weights',Weights)   # tensorflow 0.12 以下版的
              '''绘制层'''
              tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12 绘制图片
    ##and so no ……

with tf.name_scope('biases'):
    biases = tf.Variable(tf.zeros([1,out_size])+0.1, name='b')
    # tf.histogram_summary(layer_name+'/biase',biases)   # tensorflow 0.12 以下版的
    '''绘制biases'''
    tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12

# tf.histogram_summary(layer_name+'/outputs',outputs) # tensorflow 0.12 以下版本
'''绘制output'''
tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12
