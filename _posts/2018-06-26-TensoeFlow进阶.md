---
layout:     post
title:      Tensorflow进阶
subtitle:   For Deep Learning
date:       2018-06-26
catalog: true
tags:
    - Tensorflow
    - Deep Learning
---


> Last updated on 2018-10-1...

### CNN

CNN代码转载自[bryan的博客](https://blog.csdn.net/Bryan__/article/details/75452243)，任务是MNIST手写数据集的分类。

#### 数据集

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#导入数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 ```

#### 封装函数

卷积：相当于一个滤波器filter（带着一组固定权重的神经元），对局部输入数据进行卷积计算。每计算完一个数据窗口内的局部数据后，数据窗口不断平移滑动，直到计算完所有数据。

池化：即取区域平均或最大

```python
#tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。

#tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。

sess = tf.InteractiveSession()

#创建权重和偏置,初始化时应加入轻微噪声，来打破对称性，防止零梯度的问题

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#卷积使用1 步长（stride size），0 边距（padding size）的模板，保证输出和输入是同一个大小。

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化用简单传统的2X2 大小的模板做max pooling

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2,1], padding='SAME')
```

#### 搭建网络

```python
#第一层由一个卷积接一个max pooling 完成。卷积在每个5X5 的patch 中算出32 个特征。

#权重是一个[5, 5, 1, 32]的张量，前两个维度是patch 的大小，接着是输入的通道数目，最后是输出的通道数目。

#输出对应一个同样大小的偏置向量。

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#为了用这一层，我们把x变成一个4d 向量，第2、3 维对应图片的宽高，最后一维代表颜色通道。

x_image = tf.reshape(x, [-1,28,28,1])

#把x_image和权值向量进行卷积相乘，加上偏置，使用ReLU激活函数，最后maxpooling

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#为了构建一个更深的网络，把几个类似的层堆叠起来。第二层中，每个5x5的patch 会得到64 个特征。

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#现在，图片降维到7x7，加入一个有1024 个神经元的全连接层，用于处理整个图片。

#我们把池化层输出的张量reshape 成一些向量，乘上权重矩阵，加上偏置，使用ReLU 激活。

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat , W_fc1) + b_fc1)

#为了减少过拟合，我们在输出层之前加入dropout

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#最后，我们添加一个softmax 层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
```

#### 训练网络

```python
#使用交叉熵作为评估指标

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv ,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction , "float"))
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_: batch[1], keep_prob: 1.0})
        print( "step%d,training accuracy%g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})

print( "test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#使用CNN在mnist数据集上有99%的分类正确率
```

### RNN

RNN代码转载自[莫烦的课程](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)，任务是MNIST手写数据集的分类。

让 RNN 从每张图片的第一行像素读到最后一行, 然后再进行分类判断。

#### 数据集

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(1)   # set random seed

# 导入数据

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

#### 设定参数

```python
# hyperparameters

lr = 0.001                  # learning rate

training_iters = 100000     # train step 上限

batch_size = 128            
n_inputs = 28               # MNIST data input (img shape: 28*28)

n_steps = 28                # time steps

n_hidden_units = 128        # neurons in hidden layer

n_classes = 10              # MNIST classes (0-9 digits)‘

# x y placeholder

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 初始值的定义

weights = {
    # shape (28, 128)

    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),

    # shape (128, 10)

    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )

    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),

    # shape (10, )

    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
```

#### 搭建网络

这个 RNN 总共有 3 个组成部分 ( input_layer, cell, output_layer)。

```python
def RNN(X, weights, biases):

    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法

    # X ==> (128 batches * 28 steps, 28 inputs)

    X = tf.reshape(X, [-1, n_inputs])

    # X_in = W*X + b

    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维

    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # 使用 basic LSTM Cell

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) # 初始化全零 state

    # output_layer

    # 把 outputs 变成 列表 [(batch, outputs)..] * steps

    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    #选取最后一个 output

    return results
```

#### 训练网络

```python
#计算 cost 和 train_op

red = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# init= tf.initialize_all_variables() #tf马上就要废弃这种写法

# 替换成下面的写法:

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
```

#### Acc结果

```python
0.1875
0.65625
0.726562
0.757812
0.820312
0.796875
0.859375
0.921875
0.921875
0.898438
0.828125
0.890625
0.9375
0.921875
0.9375
0.929688
0.953125
....
```
