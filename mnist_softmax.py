"""
https://www.tensorflow.org/get_started/mnist/beginners
如果你已了解 MNIST 和 softmax 回归的相关知识，就可快速上手。
读完本代码你可了解 TensorFlow 的工作流程和机器学习的基本概念。
"""

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])  # 输入任意数量的MNIST图像，每张图展成784维向量；None表示第一个维度可以是任意长度
W = tf.Variable(tf.zeros([784, 10]))  # 10即0~9共10个数字
b = tf.Variable(tf.zeros([10]))  # 偏置列向量
y = tf.nn.softmax(tf.matmul(x, W) + b)  # y的大小是(m, 10)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 给占位符填入参数

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# tf.argmax 能给出某个tensor对象在某一维上的其数据最大值所在的索引值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # tf.cast 进行类型转换，tf.reduce_mean 跨越维度的计算张量各元素的平均值

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
