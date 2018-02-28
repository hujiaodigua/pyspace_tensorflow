import tensorflow as tf

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product = tf.matmul(matrix1, matrix2)  #tensorflow中的矩阵乘法。相当于numpy中的dot

#使用 Session 来激活 product 并得到计算结果
#method1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
