import tensorflow as tf
import numpy as np

#创建数据
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3 

#开始创建结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

#计算误差
loss = tf.reduce_mean(tf.square(y-y_data))
#传播误差
optimizer = tf.train.GradientDescentOptimizer(0.5) #参数是学习效率一般小于1
train = optimizer.minimize(loss)
#结束创建结构

#训练
init = tf.global_variables_initializer()  # 替换成这样就好
sess = tf.Session()
sess.run(init)          # Very important  激活网络

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
