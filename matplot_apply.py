from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
定义添加神经层函数
'''
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

##plt.scatter(x_data, y_data)
##plt.show()

#利用占位符定义我们所需的神经网络的输入，把地方先在网络中占下
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

'''
搭建网络
'''
# 隐藏层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(l1, 10, 1, activation_function=None)

#
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

#每训练一步，以0.1的效率（GradientDescentOptimizer）减小误差 （minimize）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化一个变量
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#定义Session，并用 Session 来执行 init 初始化步骤。 （注意：在tensorflow中，只有session.run()才会执行我们定义的运算。）

'''
绘制真实数据
'''
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion() #对figure执行此操作可以让plt.show一次后不暂停(阻塞)
plt.show()


'''
训练
'''
for i in range(1000):
    sess.run(train_step, feed_dict = {xs:x_data, ys:y_data})
#每50步输出一下机器学习的误差
    if i % 50 == 0:
        #可视化结果
        try:
            ax.lines.remove(lines[0]) #lines(因为lines只有一个元素，所以就是lines[0])是上一次循环中画的线，本次循环中把上一条循环的线先remove掉
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict = {xs: x_data})
        #绘制预测
        lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
        plt.pause(1)

