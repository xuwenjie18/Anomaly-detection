# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:37:42 2018

@author: xuwenjie
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
user ='ACM2278' #['ACM2278', 'CMP2946', 'PLJ1771','CDE1846','MBG3183','HIS1706']
data = np.float32(np.load(user+"_data_weekday.npy"))
x_data, y = data[:,1:], data[:,0]
x_std = sc_x.fit_transform(x_data).T

#development set and test set split
split_index = int(0.8*x_std.shape[1])
x_std_train, x_std_test  = x_std[:,:split_index], x_std[:,split_index:]

tf.reset_default_graph()
#probability density estimation
NX = x_std_train.shape[0]
#NLAYER = 2
NHIDDEN1 = 5
NHIDDEN2 = 2
STDEV = 0.5
#KMIX = 24 #number of mixtures
NOUT =  NX*2 # mu, stdev
tf.set_random_seed(1234) #to generate reproducible results

x = tf.placeholder(dtype=tf.float32, shape=[NX,None],name = 'x' )
W1 = tf.Variable(tf.random_normal([NHIDDEN1,NX], stddev = STDEV, dtype =tf.float32))
b1 = tf.Variable(tf.random_normal([NHIDDEN1,1], stddev = STDEV, dtype=tf.float32))
W2 = tf.Variable(tf.random_normal([NHIDDEN2,NHIDDEN1],stddev=STDEV,dtype=tf.float32))
b2 = tf.Variable(tf.random_normal([NHIDDEN2,1],stddev=STDEV,dtype=tf.float32))
W3 = tf.Variable(tf.random_normal([NHIDDEN1,NHIDDEN2],stddev=STDEV,dtype=tf.float32))
b3 = tf.Variable(tf.random_normal([NHIDDEN1,1],stddev=STDEV,dtype=tf.float32))
W4 = tf.Variable(tf.random_normal([NOUT,NHIDDEN1], stddev = STDEV, dtype=tf.float32))
b4 = tf.Variable(tf.random_normal([NOUT,1], stddev=STDEV, dtype=tf.float32))
hidden_layer1 = tf.nn.tanh(tf.matmul(W1,x)+b1) #NHIDDEN1 X 1
hidden_layer2 = tf.matmul(W2, hidden_layer1) + b2 #NHIDDEN2 X 1 
hidden_layer3 = tf.nn.tanh(tf.matmul(W3, hidden_layer2)+b3) #NHIDDEN1 X 1
output = tf.matmul(W4,hidden_layer3) + b4 #NOUT X 1

def get_gaussian_coef(output):
    out_mu = tf.placeholder(dtype=tf.float32, shape=[NX,None], name = 'mixparam')
    out_sigma = tf.placeholder(dtype=tf.float32, shape=[NX,None], name = "mixparam")
    
    out_mu, out_sigma = tf.split(output,axis=0,num_or_size_splits=2)
    out_sigma = tf.exp(out_sigma)
    return  out_mu, out_sigma
out_mu, out_sigma = get_gaussian_coef(output)


def get_lossfunc(out_mu, out_sigma, x):
    result = tf.subtract(x,out_mu)
    result = tf.div(result,out_sigma)
    result = tf.square(result)/2
    result = tf.add(result,tf.log(out_sigma))
    result = tf.reduce_sum(result,axis = 0)
    return result

L2regul = 0.01*(tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4))
loss_woreg = tf.reduce_mean(get_lossfunc(out_mu,out_sigma,x))
lossfunc = loss_woreg + L2regul
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

with tf.Session() as sess1:  #tf.InteractiveSession() for use in shell
    tf.set_random_seed(1234)
    sess1.run(tf.global_variables_initializer())   
    NEPOCH = 2000
    loss = np.zeros(NEPOCH)
    for i in range(NEPOCH):
        sess1.run(train_op,feed_dict={x:x_std_train})
        loss[i] = sess1.run(lossfunc, feed_dict={x: x_std_train})      
    plt.figure(figsize=(8,8))
    plt.plot(np.arange(100,NEPOCH,1),loss[100:],'r-')
    plt.title('loss versus epoch')       
    out_mu_test,out_sigma_test = sess1.run([out_mu, out_sigma],feed_dict = {x: x_std})
    anomly_score11 = get_lossfunc(out_mu_test,out_sigma_test,x_std).eval() 

    plt.figure(figsize=(8,8))
    for i in range(x_data.shape[1]):
        plt.plot(np.ones(x_data[:,i].shape)*i, x_data[:,i],'gx',alpha = 0.3)
        plt.title("x_data")
        plt.xlabel("feature")
    plt.figure(figsize=(8,8))
    for i in range(x_std.shape[0]):
        plt.plot(np.ones(x_std[i,:].shape)*i, x_std[i,:],'gx',alpha = 0.3)
        plt.title("x_std")
        plt.xlabel("feature")

    plt.figure(figsize = (8,8))
    for i in range(out_mu_test.shape[0]):
        plt.plot(np.ones(out_mu_test[i,:].shape)*i,out_mu_test[i,:],'ro',alpha = 0.3)
        plt.title("out_mu")
        plt.xlabel("feature")

    plt.figure(figsize=(8,8))
    for i in range(out_sigma_test.shape[0]):
        plt.plot(np.ones(out_sigma_test[i,:].shape)*i,out_sigma_test[i,:],'go',alpha=0.3)
        plt.title("out_sigma")
        plt.xlabel("feature")

    #plot anomly score versus day
    plt.figure(figsize=(8,8))
    plt.plot(anomly_score11,'go',alpha = 0.3)
    plt.plot([split_index,split_index],[anomly_score11.min(),anomly_score11.max()], 'r')
    plt.plot(np.where(y==1)[0],anomly_score11[np.where(y==1)],'rx')
    plt.title("anomly score for probability density estimation")
    plt.xlabel("day")

    plt.grid(True)
    plt.show()
    index = np.where(anomly_score11> -50)[0]
    index = np.append(index,0)
    Anomly = np.square(np.divide(x_std-out_mu_test,out_sigma_test))*0.5+ np.log(out_sigma_test)
    for item in index: # plot the anomly score for each feature
        plt.figure(figsize=(8,8))
        plt.stem(Anomly[:,item])
        plt.title("Anomly score component for day "+str(item))
        plt.xlabel("day")
        plt.grid(True)
        plt.show()
        print('day '+str(item))
        print(x_data[item,:])