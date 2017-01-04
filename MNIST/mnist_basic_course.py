# _._ coding:utf8
import get_datasource
import tensorflow as tf

# 从网上下下来的demo数据
mnist = get_datasource.mnist

# N*784的二维数组，N是行，784是列，用于放N个28*28的数组，把28*28的数组降纬处理就是784,作为占位符等待输入
x = tf.placeholder(tf.float32, [None, 784])

# 784*10的二维数组，W是28*28图片权重数组，因为只有0～9，所以只有10，这里作为运算结果的缓存，在下面的代码中循环渐渐修正这个值
# 初始化0向量
W = tf.Variable(tf.zeros([784, 10]))

# b是偏移量，这里作为运算结果的缓存，在下面的代码中循环渐渐修正这个值
# 初始化0向量
b = tf.Variable(tf.zeros([10]))

# softmax 作为激励函数（详情：https://www.zhihu.com/question/22334626/answer/21036590）或者链接（link）函数（不知道是什么）使用。
# 激励函数，使其符合线性分布
# 结果是N*10的数组
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 用于计算信息熵 -y_*log(y)，y_是图片的label，训练数据的正确答案
y_ = tf.placeholder("float", [None, 10])

# 正确的概率分布y_和估算（错误）的概率分布相乘（注意不是数组相乘）得到交叉熵
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降算法，0.01为梯度逐渐找更优解
tran_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化变量
init = tf.initialize_all_variables()

# 初始化变量,
session = tf.Session()
session.run(init)

for i in range(1000):
    # batch_xs是图像数字化后得到的28*28的数组x，batch_yx是正确分布y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(tran_step, feed_dict={x: batch_xs, y_: batch_ys})
    if i% 50 == 0:
        # 观察W，b的变化
        session.run(tf.Print(W, [W], 'W: '))
        session.run(tf.Print(b, [b], 'b: '))

        # 因为y和y_都是N*10的数组，所以用第一纬度，也就是10这纬度中的最大值（预测是哪个数字）对比，测试正确为1，错误为0
        # correct_prediction 是一个一维数组，长度等于输入变量长度
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

        # accuracy总和求平均
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print "Step: ", i, "Accuracy: ", session.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels})


# 模型评估
# correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print session.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels})