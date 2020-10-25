import tensorflow as tf
import numpy as np
import sys
import time
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
from pdADMM import common
from pdADMM.input_data import mnist,fashion_mnist,kmnist,svhn,cifar10,cifar100

def Net(images, label, num_of_neurons):
    data =np.load("pdadmm_cifar100_full_batch_1000_iter_1e-4_lr_500_5layers.npz")
    W1 =tf.Variable(data["arr_0"])
    b1=tf.Variable(data["arr_1"])
    z1=tf.Variable(data["arr_2"])
    q1=tf.Variable(data["arr_3"])
    p2=tf.Variable(data["arr_4"])
    W2=tf.Variable(data["arr_5"])
    b2=tf.Variable(data["arr_6"])
    z2=tf.Variable(data["arr_7"])
    q2=tf.Variable(data["arr_8"])
    p3=tf.Variable(data["arr_9"])
    W3=tf.Variable(data["arr_10"])
    b3=tf.Variable(data["arr_11"])
    z3=tf.Variable(data["arr_12"])
    q3=tf.Variable(data["arr_13"])
    p4=tf.Variable(data["arr_14"])
    W4=tf.Variable(data["arr_15"])
    b4=tf.Variable(data["arr_16"])
    z4=tf.Variable(data["arr_17"])
    q4=tf.Variable(data["arr_18"])
    p5=tf.Variable(data["arr_19"])
    W5 = tf.Variable(tf.eye(num_of_neurons, num_of_neurons))
    b5 = tf.Variable(tf.zeros(shape=(num_of_neurons, 1)))
    z5 = tf.matmul(W5, p5) + b5
    q5 = common.relu(z5)
    p6 = q5
    W6 = tf.Variable(tf.eye(num_of_neurons, num_of_neurons))
    b6 = tf.Variable(tf.zeros(shape=(num_of_neurons, 1)))
    z6 = tf.matmul(W6, p6) + b6
    q6 = common.relu(z6)
    p7 = q6
    W7 = tf.Variable(tf.eye(num_of_neurons, num_of_neurons))
    b7 = tf.Variable(tf.zeros(shape=(num_of_neurons, 1)))
    z7 = tf.matmul(W7, p7) + b7
    q7 = common.relu(z7)
    p8 = q7
    W8 = tf.Variable(tf.eye(num_of_neurons, num_of_neurons))
    b8 = tf.Variable(tf.zeros(shape=(num_of_neurons, 1)))
    z8 = tf.matmul(W8, p8) + b8
    q8 = common.relu(z8)
    p9 = q8
    W9 = tf.Variable(tf.eye(num_of_neurons, num_of_neurons))
    b9 = tf.Variable(tf.zeros(shape=(num_of_neurons, 1)))
    z9 = tf.matmul(W9, p9) + b9
    q9 = common.relu(z9)
    p10 = q9
    W10 = tf.Variable(tf.eye(100, num_of_neurons))
    b10 = tf.Variable(tf.zeros(shape=(100, 1)))
    imask = tf.equal(label, 0)
    z10 = tf.where(imask, -tf.ones_like(label), tf.ones_like(label))
    return W1,b1,z1,q1,p2,W2,b2,z2,q2,p3,W3,b3,z3,q3,p4,W4,b4,z4,q4,p5,W5,b5,z5,q5,p6,W6,b6,z6,q6,p7,W7,b7,z7,q7,p8,W8,b8,z8,q8,p9,W9,b9,z9,q9,p10,W10,b10,z10

# return the accuracy of the neural network model
def test_accuracy(W1, b1, W2, b2, W3, b3,W4,b4,W5,b5,W6,b6,W7,b7,W8,b8,W9,b9,W10,b10, images, labels):
    nums = int(labels.shape[1])
    z1 = tf.matmul(W1, images) + b1
    q1 = common.relu(z1)
    p2 = q1
    z2 = tf.matmul(W2, q1) + b2
    q2 = common.relu(z2)
    p3 = q2
    z3 = tf.matmul(W3, p3) + b3
    q3 = common.relu(z3)
    p4 = q3
    z4 = tf.matmul(W4, p4) + b4
    q4 = common.relu(z4)
    p5 = q4
    z5 = tf.matmul(W5, p5) + b5
    q5 = common.relu(z5)
    p6 = q5
    z6 = tf.matmul(W6, p6) + b6
    q6 = common.relu(z6)
    p7 = q6
    z7 = tf.matmul(W7, p7) + b7
    q7 = common.relu(z7)
    p8 = q7
    z8 = tf.matmul(W8, p8) + b8
    q8 = common.relu(z8)
    p9 = q8
    z9 = tf.matmul(W9, p9) + b9
    q9 = common.relu(z9)
    p10 = q9
    z10 = tf.matmul(W10, p10) + b10
    cost = common.cross_entropy_with_softmax(labels, z10) / nums
    label = tf.argmax(labels, axis=0)
    pred = tf.argmax(z10, axis=0)
    return (tf.reduce_sum(tf.cast(tf.equal(pred, label),tf.float32)) / nums,cost)

mnist = cifar100()
#initialization
x_train = mnist.x_train_down_sample.astype(np.float32)
y_train = mnist.y_train.astype(np.float32)
x_train = tf.transpose(x_train)
y_train = tf.transpose(y_train)
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train,dtype=tf.float32)
x_test = mnist.x_test_down_sample.astype(np.float32)
y_test = mnist.y_test.astype(np.float32)
x_test = tf.transpose(x_test)
y_test = tf.transpose(y_test)
x_test = tf.convert_to_tensor(x_test,dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test,dtype=tf.float32)
num_of_neurons = 500
ITER = 50
index = 0
W1, b1, z1, q1, p2, W2, b2, z2, q2, p3, W3, b3, z3, q3, p4, W4, b4, z4, q4, p5, W5, b5, z5, q5, p6, W6, b6, z6, q6, p7, W7, b7, z7, q7, p8, W8, b8, z8, q8, p9, W9, b9, z9, q9, p10, W10, b10, z10= Net(x_train, y_train, num_of_neurons)
u1=tf.zeros(q1.shape)
u2=tf.zeros(q2.shape)
u3=tf.zeros(q3.shape)
u4=tf.zeros(q4.shape)
u5=tf.zeros(q5.shape)
u6=tf.zeros(q6.shape)
u7=tf.zeros(q7.shape)
u8=tf.zeros(q8.shape)
u9=tf.zeros(q9.shape)
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
    p6 =common.update_p(p6,q5,W6,b6,z6,u5,rho)
    p7 =common.update_p(p7,q6,W7,b7,z7,u6,rho)
    p8 =common.update_p(p8,q7,W8,b8,z8,u7,rho)
    p9 =common.update_p(p9,q8,W9,b9,z9,u8,rho)
    p10 =common.update_p(p10,q9,W10,b10,z10,u9,rho)
    W1 =common.update_W(x_train,b1,z1,W1,rho)
    W2 =common.update_W(p2,b2,z2,W2,rho)
    W3 =common.update_W(p3,b3,z3,W3,rho)
    W4 =common.update_W(p4,b4,z4,W4,rho)
    W5 =common.update_W(p5,b5,z5,W5,rho)
    W6 =common.update_W(p6,b6,z6,W6,rho)
    W7 =common.update_W(p7,b7,z7,W7,rho)
    W8 =common.update_W(p8,b8,z8,W8,rho)
    W9 =common.update_W(p9,b9,z9,W9,rho)
    W10 =common.update_W(p10,b10,z10,W10,rho)
    b1 =common.update_b(x_train,W1,z1,b1,rho)
    b2 =common.update_b(p2,W2,z2,b2,rho)
    b3 =common.update_b(p3,W3,z3,b3,rho)
    b4 =common.update_b(p4,W4,z4,b4,rho)
    b5 =common.update_b(p5,W5,z5,b5,rho)
    b6 =common.update_b(p6,W6,z6,b6,rho)
    b7 =common.update_b(p7,W7,z7,b7,rho)
    b8 =common.update_b(p8,W8,z8,b8,rho)
    b9 =common.update_b(p9,W9,z9,b9,rho)
    b10 =common.update_b(p10,W10,z10,b10,rho)
    z1 =common.update_z(z1,x_train,W1,b1,q1,rho)
    z2 =common.update_z(z2,p2,W2,b2,q2,rho)
    z3 =common.update_z(z3,p3,W3,b3,q3,rho)
    z4 =common.update_z(z4,p4,W4,b4,q4,rho)
    z5 =common.update_z(z5,p5,W5,b5,q5,rho)
    z6 =common.update_z(z6,p6,W6,b6,q6,rho)
    z7 =common.update_z(z7,p7,W7,b7,q7,rho)
    z8 =common.update_z(z8,p8,W8,b8,q8,rho)
    z9 =common.update_z(z9,p9,W9,b9,q9,rho)
    z10 =common.update_zl(p10,W10,b10,y_train,z10,rho)
    q1 =common.update_q(p2,z1,u1,rho)
    q2 =common.update_q(p3,z2,u2,rho)
    q3 =common.update_q(p4,z3,u3,rho)
    q4 =common.update_q(p5,z4,u4,rho)
    q5 =common.update_q(p6,z5,u5,rho)
    q6 =common.update_q(p7,z6,u6,rho)
    q7 =common.update_q(p8,z7,u7,rho)
    q8 =common.update_q(p9,z8,u8,rho)
    q9 =common.update_q(p10,z9,u9,rho)
    u1 = u1 +rho*(p2-q1)
    u2 = u2 +rho*(p3-q2)
    u3 = u3 +rho*(p4-q3)
    u4 = u4 +rho*(p5-q4)
    u5 = u5 +rho*(p6-q5)
    u6 = u6 +rho*(p7-q6)
    u7 = u7 +rho*(p8-q7)
    u8 = u8 +rho*(p9-q8)
    u9 = u9 +rho*(p10-q9)
    print("Time per iteration:", time.time() - pre)
    print("rho=",rho)
    (train_acc[i],train_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3,W4, b4, W5, b5, W6, b6, W7, b7,W8, b8, W9, b9, W10, b10, x_train,y_train)
    print("training cost:", train_cost[i])
    print("training acc:",train_acc[i])
    (test_acc[i],test_cost[i])=test_accuracy(W1, b1, W2, b2, W3, b3,W4, b4, W5, b5, W6, b6, W7, b7,W8, b8, W9, b9, W10, b10, x_test, y_test)
    print("test cost:", test_cost[i])
    print("test acc:",test_acc[i])
    np.savez('pdadmm_cifar100_full_batch_100_iter_1e-4_lr_'+repr(num_of_neurons), train_acc, test_acc)