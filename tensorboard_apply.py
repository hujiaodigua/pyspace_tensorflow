from __future__ import print_function
import tensorflow as tf
import numpy as np

'''
定义添加神经层函数
'''
def add_layer(inputs, in_size, out_size, activation_function = None):
    with tf.name_scope('weights'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = 'W')
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

#利用占位符定义我们所需的神经网络的输入，把地方先在网络中占下
xs = tf.placeholder(tf.float32, [None, 1], name = 'x_in')
ys = tf.placeholder(tf.float32, [None, 1], name = 'y_in')

'''
搭建网络
'''
# 隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

#
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

#每训练一步，以0.1的效率（GradientDescentOptimizer）减小误差 （minimize）
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

# tf.train.SummaryWriter soon be deprecated, use following
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:  # tensorflow version < 0.12
    writer = tf.train.SummaryWriter('logs/', sess.graph)
else: # tensorflow version >= 0.12
    writer = tf.summary.FileWriter("logs/", sess.graph)

#初始化一个变量
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
#定义Session，并用 Session 来执行 init 初始化步骤。 （注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）

# '''
# 训练
# '''
# for i in range(1000):
#     sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
# #每50步输出一下机器学习的误差
#     if i % 50 == 0:
#         print(sess.run(loss, feed_dict = {xs:x_data, ys:y_data}))
