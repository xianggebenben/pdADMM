import tensorflow as tf
import numpy as np

def cross_entropy_with_softmax(label, zl):
    prob = softmax(zl)
    imask =tf.equal(prob,0.0)
    prob = tf.where(imask,1e-10,prob)
    loss = cross_entropy(label, prob)
    return loss
def softmax(x):
    exp =tf.math.exp(x)
    imask =tf.equal(exp,float("inf"))
    exp = tf.where(imask,tf.math.exp(88.6),exp)
    return exp/(tf.math.reduce_sum(exp,axis=0)+1e-10)
def cross_entropy(label, prob):
    loss = -tf.math.reduce_sum(label * tf.math.log(prob))
    return loss
#return the  relu function
def relu(x):
    return tf.maximum(x, 0)
# return phi
def eq1(p, W, b, z, rho):
    temp = z - tf.matmul(W, p) - b
    res = rho / 2 * tf.reduce_sum(temp * temp)
    return res
# return the derivative of phi with regard to W
def eq1_W(p, W, b, z,rho):
    temp = tf.matmul(W, p) + b - z
    temp2 = tf.transpose(p)
    res = rho * tf.matmul(temp, temp2)
    return res
# return the derivative of phi with regard to b
def eq1_b(p, W, b, z, rho):
    res = tf.reshape(tf.reduce_mean(rho * (tf.matmul(W, p) + b - z), axis=1),shape=(-1, 1))
    return res
# return the derivative of phi with regard to z
def eq1_z(a, W, b, z, rho):
    res = rho * (z - b - tf.matmul(W, a))
    return res
# return the quadratic approximation of W-subproblem
def P(W_new, theta, p, W, b, z,rho):
    temp = W_new - W
    res = eq1(p, W, b, z,rho) + tf.reduce_sum(eq1_W(p, W, b, z,rho) * temp) + tf.reduce_sum(theta * temp * temp) / 2
    return res
# return the quadratic approximation of p-subproblem
def Q(p_new, tau, p, q, W, b, z, u,gradient,rho):
    temp = p_new - p
    res = p_obj(p, q,W, b, z, u,rho) + tf.reduce_sum(gradient * temp) + tf.reduce_sum(
        tau * temp * temp) / 2
    return res
def p_obj(p,q,W,b,z,u,rho):
    f =rho/2*tf.reduce_sum((z-tf.matmul(W,p)-b)*(z-tf.matmul(W,p)-b))+tf.linalg.trace(tf.matmul(p-q,tf.transpose(u)))+rho/2*tf.reduce_sum((p-q)*(p-q))
    return f
def update_p(p_old,q,W, b, z,u,rho):
    with tf.GradientTape() as g:
        g.watch(p_old)
        f=p_obj(p_old, q,W, b, z, u,rho)
    gradient =g.gradient(f,p_old)
    eta = 2
    t=10
    beta=p_old-gradient/t
    while (p_obj(beta, q,W, b, z, u,rho) > Q(beta, t, p_old, q, W, b, z, u,gradient,rho)):
        t = t * eta
        beta=p_old-gradient/t
    tau = t
    p = beta
    return p
# return the result of W-subproblem
def update_W(p, b, z, W_old, rho):
    gradients = eq1_W(p, W_old, b, z, rho)
    gamma = 2
    alpha = 10
    zeta = W_old - gradients / alpha
    while (eq1(p, zeta, b, z, rho) > P(zeta, alpha, p, W_old, b, z,rho)):
        alpha = alpha * gamma
        zeta = W_old - gradients / alpha  # Learning rate decreases to 0, leading to infinity loop here.
    theta = alpha
    W = zeta
    return W
# return the result of b-subproblem
def update_b(p, W, z, b_old, rho):
    gradients = eq1_b(p, W, b_old, z, rho)
    res = b_old - gradients / rho
    return res
# return the objective value of z-subproblem
def z_obj(p, W, b, z, q,rho):
    f=rho/2*(z-tf.matmul(W,p)-b)*(z-tf.matmul(W,p)-b)+rho/2*(q-relu(z))*(q-relu(z))
    return f
# return the result of z-subproblem
def update_z(z,p, W, b, q, rho):
    z1=(tf.matmul(W,p)+b+z)/2;
    z2=(2*z1+q)/3
    z1=tf.minimum(z1,0)
    z2=tf.maximum(z2,0)
    value1=z_obj(p, W, b, z1, q,rho)
    value2=z_obj(p, W, b, z2, q,rho)
    imask =tf.math.greater(value1, value2)
    z=tf.where(imask, z2,z1)
    return z
# return the result of z_L-subproblem by FISTA
def update_zl(p, W, b, label, zl_old, rho):
    fzl = 10e10
    MAX_ITER = 500
    zl = zl_old
    lamda = 1
    zeta = zl
    eta = 4
    TOLERANCE = 1e-3
    for i in range(MAX_ITER):
        fzl_old = fzl
        fzl = cross_entropy_with_softmax(label, zl)+rho/2*tf.reduce_sum((zl-tf.matmul(W,p)-b)*(zl-tf.matmul(W,p)-b))
        if abs(fzl - fzl_old) < TOLERANCE:
            break
        lamda_old = lamda
        lamda = (1 + np.sqrt(1 + 4 * lamda * lamda)) / 2
        gamma = (1 - lamda_old) / lamda
        gradients2 = (softmax(zl) - label)
        zeta_old = zeta
        zeta = (rho * (tf.matmul(W, p)+b) + (zl - eta * gradients2) / eta) / (rho + 1 / eta)
        zl = (1 - gamma) * zeta + gamma * zeta_old
    return zl
def update_q(p,z,u,rho):
    res =(p+u/rho+relu(z))/2
    return res