---
layout:     post
title:      Tensorflow基础
subtitle:   For practice
date:       2018-06-26
catalog: true
tags:
    - Tensorflow
    - Deep Learning
---


> Last updated on 2019-6-19...  

### 计算图的概念

计算图是一个强大的工具，绝大部分神经网络都可以用计算图描述。

**计算图用节点表示变量（标量、向量、矩阵、张量都可以），用有向边表示计算。**

自动求导应用链式法则求某节点对其他节点的雅可比矩阵，它从结果节点开始，沿着计算路径向前追溯，**逐节点计算雅可比**。

将神经网络和损失函数连接成一个计算图，则它的输入、输出和参数都是节点，可利用自动求导求损失值对网络参数的雅可比，从而得到梯度。

> [《计算图反向传播的原理及实现》](https://mp.weixin.qq.com/s/KCCsTQ87BThVDcZcAfC68Q)

### 引入头文件

```python
import tensorflow as tf
```

### 常量&变量

```python
# 定义常量 a

a = tf.constant(1)

# 定义变量 b（这个例子中把b作计数器了）

b = tf.Variable(0, name='counter')
print(b.name)

# 加法（等式左边在tf中可以理解为步骤名）

sum = tf.add(a,b)

# 更新赋值

update = tf.assign(b, sum)
```
### 初始化所有变量并且运行（激活）

```python
# 如果定义了变量，就一定要初始化它们

init = tf.global_variables_initializer()

# 运行（把sess理解为指针，sess.run(xx步骤)代表运行该步骤）

with tf.Session() as sess:
    sess.run(init)             #运行变量的初始化

    for _ in range(3):
        sess.run(update)       #运行update步骤

        print(sess.run(sum))   #运行sum步骤
```

### 矩阵

```python

#定义矩阵

matrix1 = tf.constant([ [3., 3.] ])   #1x2的矩阵

matrix2 = tf.constant([ [2.],         #2x1的矩阵

                        [2.]  ])

#矩阵乘法

product = tf.matmul(matrix1, matrix2)
```

### 运行（即启动图）

```python
#方法1

sess = tf.Session()
result = sess.run(product)
print result
sess.close()

#方法2

with tf.Session() as sess：
    result = sess.run([product])
    print result
```

### 指派GPU运行

```python
#通常情况下，你不需要显示指使用CPU或者GPU。TensorFlow能自动检测，如果检测到GPU，TensorFlow会使用第一个GPU来执行操作。
#如果机器上有多个GPU，除第一个GPU外的其他GPU是不参与计算的，为了使用这些GPU，你必须将op明确指派给他们执行。
#with…Device语句用来指派特定的CPU或GPU执行操作：

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1, matrix2)
        ...
```

### Feed机制（命令行输入参数）

```python
#之前的例子中展示了在计算图中引入tensor，以常量和变量的形式存储。
#TensorFlow还提供了feed机制，该机制可以临时替换图中的tensor。

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print sess.run([output], feed_dict={input1:[7.], input2:[2.]})

# 输出:
# [array([ 14.], dtype=float32)]
```

### 命令行参数（Flag）

详见[博客1](https://blog.csdn.net/u012436149/article/details/52870069)和[博客2](https://blog.csdn.net/lyc_yongcai/article/details/73456960)

#### 新建test.py文件：

```python
#定义参数

flags = tf.flags
flags.DEFINE_string("para_name_1","default_val", "description")
flags.DEFINE_bool("para_name_2","default_val", "description")

#session主体


def main(_):
    ...
```

#### 命令行调用

```python
python test.py --para_name_1=name --para_name_2=name2
```
