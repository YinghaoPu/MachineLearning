from get_datasource import mnist
import tensorflow as tf

# InteractiveSession类，通过它，你可以更加灵活地构建你的代码。它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
session = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stdev=0.1)
    return tf.Variable(initial)

def bais_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")



W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bais_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bais_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bais_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bais_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_conv2) + b_fc2)
cross_entropy = tf.reduce_sum(y_ * tf.log(y_conv))
tran_step = tf.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.cast(tf.argmax(y_conv, 1), tf.argmax(y_, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
session.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print "step %d, training accuracy &g" % (i, train_accuracy)
    tran_step.run(feed_dict={x:batch[0], y_:batch[0], keep_prob:0.5})

print  "test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
