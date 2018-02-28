import tensorflow as tf

'''
在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。
'''

#tensorflow定义变量
state = tf.Variable(0, name = 'counter')
print(state.name)

#tensorflow定义常量 one
one = tf.constant(1)

#定义加法步骤（注：此步并没有直接计算）
new_value = tf.add(state, one)

#将State更新成new_value
update = tf.assign(state, new_value)

# 如果定义 Variable, 就一定要 initialize
init = tf.global_variables_initializer()

#使用Session
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))#一定要把state放到session上面去run一下就把 sess 的指针指向 state
