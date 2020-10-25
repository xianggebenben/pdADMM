import tensorflow as tf
import numpy as np
import sys
import time
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from pdADMM import common
from pdADMM.input_data import mnist,fashion_mnist,kmnist,svhn,cifar10,cifar100


# initialize the neural network
def Net(images, label, num_of_neurons):
    seed_num = 0
    tf.random.set_seed(seed=seed_num)
    W1 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 16 * 16 * 3),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b1 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z1 = tf.matmul(W1, images) + b1
    q1 = common.relu(z1)
    p2 = q1
    tf.random.set_seed(seed=seed_num)
    W2 = tf.Variable(tf.random.normal(shape=(num_of_neurons, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b2 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z2 = tf.matmul(W2, q1) + b2
    q2 = common.relu(z2)
    p3 = q2
    tf.random.set_seed(seed=seed_num)
    W3 = tf.Variable(tf.random.normal(shape=(num_of_neurons, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b3 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z3 = tf.matmul(W3, p3) + b3
    q3 = common.relu(z3)
    p4 = q3
    tf.random.set_seed(seed=seed_num)
    W4 = tf.Variable(tf.random.normal(shape=(num_of_neurons, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b4 = tf.Variable(tf.random.normal(shape=(num_of_neurons, 1),mean=0,stddev=0.1))
    z4 = tf.matmul(W4, p4) + b4
    q4 = common.relu(z4)
    p5 = q4
    tf.random.set_seed(seed=seed_num)
    W5 = tf.Variable(tf.random.normal(shape=(100, num_of_neurons),mean=0,stddev=0.1))
    tf.random.set_seed(seed=seed_num)
    b5 = tf.Variable(tf.random.normal(shape=(100, 1),mean=0,stddev=0.1))
    imask = tf.equal(label, 0)
    z5 = tf.where(imask, -tf.ones_like(label), tf.ones_like(label))
    return W1,b1,z1,q1,p2,W2,b2,z2,q2,p3,W3,b3,z3,q3,p4,W4,b4,z4,q4,p5,W5,b5,z5

# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3,W4,b4,W5,b5, images, labels):
    nums = int(labels.shape[1])
    z1 = tf.matmul(W1, images) + b1
    q1 = common.relu(z1)
    p2 = q1
    z2 = tf.matmul(W2, p2) + b2
    q2 = common.relu(z2)
    p3 = q2
    z3 = tf.matmul(W3, p3) + b3
    q3 = common.relu(z3)
    p4 = q3
    z4 = tf.matmul(W4, p4) + b4
    q4 = common.relu(z4)
    p5 = q4
    z5 = tf.matmul(W5, p5) + b5
    cost = common.cross_entropy_with_softmax(labels, z5) / nums
    label = tf.argmax(labels, axis=0)
    pred = tf.argmax(z5, axis=0)
    return (tf.reduce_sum(tf.cast(tf.equal(pred, label),tf.float32)) / nums,cost)

mnist = cifar100()
#initialization
x_train = mnist.train_down_sample.xs.astype(np.float32)
y_train = mnist.train_down_sample.ys.astype(np.float32)
x_train = tf.transpose(x_train)
y_train = tf.transpose(y_train)
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
x_test = mnist.test_down_sample.xs.astype(np.float32)
y_test = mnist.test_down_sample.ys.astype(np.float32)
x_test = tf.transpose(x_test)
y_test = tf.transpose(y_test)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)
num_of_neurons = 500
ITER = 50
index = 0
W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5= Net(x_train, y_train, num_of_neurons)
u1=tf.zeros(q1.shape)
u2=tf.zeros(q2.shape)
u3=tf.zeros(q3.shape)
u4=tf.zeros(q4.shape)
train_acc=np.zeros(ITER)
test_acc=np.zeros(ITER)
linear_r=np.zeros(ITER)
objective_value=np.zeros(ITER)
train_cost=np.zeros(ITER)
test_cost=np.zeros(ITER)
rho = 1e-4
for i in range(ITER):
    pre = time.time()
    print("iter=",i)
    p2 =common.update_p(p2,q1,W2,b2,z2,u1,rho)
    p3 =common.update_p(p3,q2,W3,b3,z3,u2,rho)
    p4 =common.update_p(p4,q3,W4,b4,z4,u3,rho)
    p5 =common.update_p(p5,q4,W5,b5,z5,u4,rho)
    W1 =common.update_W(x_train,b1,z1,W1,rho)
    W2 =common.update_W(p2,b2,z2,W2,rho)
    W3 =common.update_W(p3,b3,z3,W3,rho)
    W4 =common.update_W(p4,b4,z4,W4,rho)
    W5 =common.update_W(p5,b5,z5,W5,rho)
    b1 =common.update_b(x_train,W1,z1,b1,rho)
    b2 =common.update_b(p2,W2,z2,b2,rho)
    b3 =common.update_b(p3,W3,z3,b3,rho)
    b4 =common.update_b(p4,W4,z4,b4,rho)
    b5 =common.update_b(p5,W5,z5,b5,rho)
    z1 =common.update_z(z1,x_train,W1,b1,q1,rho)
    z2 =common.update_z(z2,p2,W2,b2,q2,rho)
    z3 =common.update_z(z3,p3,W3,b3,q3,rho)
    z4 =common.update_z(z4,p4,W4,b4,q4,rho)
    z5 =common.update_zl(p5,W5,b5,y_train,z5,rho)
    q1 =common.update_q(p2,z1,u1,rho)
    q2 =common.update_q(p3,z2,u2,rho)
    q3 =common.update_q(p4,z3,u3,rho)
    q4 =common.update_q(p5,z4,u4,rho)
    u1 = u1 +rho*(p2-q1)
    u2 = u2 +rho*(p3-q2)
    u3 = u3 +rho*(p4-q3)
    u4 = u4 +rho*(p5-q4)
    print("Time per iteration:", time.time() - pre)
    print("rho=",rho)
    (train_acc[i],train_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3,W4, b4, W5, b5, x_train,y_train)
    print("training cost:", train_cost[i])
    print("training acc:",train_acc[i])
    (test_acc[i],test_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3,W4, b4, W5, b5, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:",test_acc[i])
np.savez('pdadmm_cifar100_full_batch_1000_iter_1e-4_lr_'+repr(num_of_neurons)+'_5layers', W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5)