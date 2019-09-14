# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:25:08 2019

@author: Mariano L. Acosta
"""

import numpy as np
import tensorflow as tf
import random as rnd

class FCNN:
    def __init__(self,net,X,Y,w,b):
        self.net = net
        self.X = X
        self.Y = Y
        self.weights = w
        self.biases = b
    
    def print_w(self,sess):
        i=0
        for w in self.weights:
            print('Layer #' + str(i)+':')
            print(w.eval(session=sess))
            i+=1
    
    def predict(self,x,sess, output = 'regression' ):
        
        if output == 'regression':
            data = sess.run(self.net, feed_dict={self.X:x})
            
        elif output == "class":
            
            data = sess.run(tf.nn.softmax(tf.transpose(self.net)), feed_dict={self.X:x})
            data = np.transpose(data)
        return data
    
    def saveModel(self,sess,name):
        saver = tf.train.Saver()
        saver.save(sess,name)
        
        return saver
    
    def loadModel():
        pass
                    
def one_hot_matrix(labels):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    n_values = np.max(labels) + 1
    
    if n_values == 1:
        n_values = 2
        
    one_hot = np.transpose(np.eye(n_values)[labels.tolist()])

    
    return one_hot

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of input vector 
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32,shape =(n_x, None),name ="X")
    Y = tf.placeholder(tf.float32,shape = (n_y,None),name ="Y" )
    
    return X, Y

def create_parameters(layer,size_prev,size):
    """
    Creates and initializes parameters to build a neural network with tensorflow.
    Arguments:
        layer -- scalar, current layer
        size_prev -- scalar, number of input elements
        size_prev -- scalar, number of output elements
        
    Returns:
    parameters -- two tensorflow variables (weight and bias)
    """     

    W = tf.get_variable("W" + str(layer), [size,size_prev], initializer = tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b"+ str(layer), [size,1], initializer = tf.zeros_initializer())

    
    return W,b

def concat_layer(layer,neurons,Input,activation,dropout = False, drop_rate = .8):
    """
    Adds a fully connected layer to a previous tensorflow grap
    Arguments:
        layer -- scalar, current layer
        neurons -- list, number of neuron per layer
        Input -- graph it could be a placeholder or a previous neural layer
        
    Returns:
        A -- two tensorflow variables (weight and bias)
    """      
    size_prev = Input.get_shape().as_list()[0]
    W,b = create_parameters(layer,size_prev,neurons)
   
    Z = tf.matmul(W,Input) + b  
   
    if activation == 'relu':
        A = tf.nn.relu(Z)
    elif activation == 'leaky_relu':
        A = tf.nn.leaky_relu(Z) 
    elif activation =='tanh':
        A = tf.tanh (Z)
    elif activation == 'sigmoid':
        A = tf.sigmoid(Z)
    elif activation == 'linear':
        return Z, W, b
    elif activation == 'softmax':
        A = tf.softmax(Z)
    else: 
        raise Exception('Unknown name for activation function')
        
    if dropout == True:
        A = tf.nn.dropout(A,rate = 1- drop_rate)
        
    return A, W, b


def create_FCNN(X_dim,Y_dim,neurons_per_layer,activation, dropout = False , drop_rate = .5):
    
    tf.reset_default_graph()
    
    Wl = []
    Bl = []
    
    X,Y = create_placeholders(X_dim, Y_dim)
    
    depth = len(neurons_per_layer)
    A = X
    
    for i in range(depth):
        
        A, W, b = concat_layer(i,neurons_per_layer[i],A,activation, dropout = dropout,
                               drop_rate = drop_rate)
        
        Wl.append(W)
        Bl.append(b)
        
    Net, W, b = concat_layer(depth,Y_dim,A,'linear')
    Wl.append(W)
    Bl.append(b)
    
    return FCNN(Net,X,Y,Wl,Bl) 


def startNN():
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess

    
def train_model(data_train, data_test, NeuralNet, optimizer ='Adam', cost_function = 'RMS', 
                Epochs = 100, learning_rate = 0.01, print_cost = False, cost_step = 1,
                L2_Reg = False, beta = 0.01, print_weight = False, minibatch_size = 1000):
        
    X = NeuralNet.X
    Y = NeuralNet.Y
    Weights = NeuralNet.weights
    Biases = NeuralNet.biases
    Net = NeuralNet.net
    
    costPerIteration = []
    (X_Train,Y_Train) = data_train
    (X_Test,Y_Test) = data_test
    
    
    if cost_function == 'cross_entropy':
        
        logits = tf.transpose(Net)
        labels = tf.transpose(Y)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels =labels))
        
    elif cost_function == 'RMS':
        
        cost = tf.reduce_mean(tf.losses.mean_squared_error(Y, Net))
        
    else: 
        raise Exception('Unknown name for cost function')
    
    
    
    if L2_Reg == True:
        regularizer = tf.Variable(tf.zeros([1, 1]), name="regularizer")
        regularizer = addRegularization(Weights,beta)
        cost += regularizer
        
    if optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    elif optimizer == 'Gradient':
        opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)   
    elif optimizer == 'RMSprop':
        opt = tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)  
    else:
        raise Exception('Unknown name for optimizer')
    

    sess = startNN()
        
    
    minibatches, num_minibatches = random_mini_batches(X_Train, Y_Train, minibatch_size)
    
    for i in range(Epochs):  
        
        epoch_cost = 0
        
        for minibatch in minibatches:
    
            (minibatch_X, minibatch_Y) = minibatch
            _ , minibatch_cost = sess.run([opt,cost],feed_dict={X:minibatch_X, Y:minibatch_Y})
            
            epoch_cost += minibatch_cost / num_minibatches
            
        
        
        if print_weight == True:
            print ("Weights after epoch %i:\n" % i)
            NeuralNet.print_w(sess)
            print('\n')
        
        if print_cost == True and i % cost_step ==0:
            costPerIteration.append(epoch_cost)
            test_cost = sess.run(cost,feed_dict={X:X_Test, Y:Y_Test})  
            print ("Cost after epoch %i: %f, validation cost: %f" % (i, epoch_cost,test_cost))
            
            

    
    test_cost = sess.run(cost,feed_dict={X:X_Test, Y:Y_Test})  
    print("\n Train error: %f" % epoch_cost )
    print("Test error: %f" % test_cost )
    return FCNN(Net,X,Y,Weights,Biases),sess, costPerIteration
    
    
    
    
def addRegularization(W,beta):
    
    f = []
    for w in W:
        f.append(tf.nn.l2_loss(w))
    
    return tf.reduce_mean(sum(f))*beta/2  
    
    

def random_mini_batches(X_train, Y_train, minibatch_size):
    
    mini_batches = []
    
    size = X_train.shape[1]
    dim = X_train.shape[0]
    dim_y = Y_train.shape[0]
    
    index = rnd.sample(range(size),size)
    
    num_of_batches = size // minibatch_size
    
    flag = 0
    
    for i in range(num_of_batches):
        
        X = np.zeros((dim,minibatch_size))
        Y = np.zeros((dim_y,minibatch_size))
        
        for j in range(minibatch_size):
            
            a = np.reshape(X_train[:,index[flag]],(dim))
            b = np.reshape(Y_train[:,index[flag]],(dim_y))
            X[:,j] = a 
            Y[:,j] = b
            flag+=1
        
        mini_batches.append([X,Y])
                
    
    return mini_batches,num_of_batches


def create_train_test(X,Y,train_percentage = 0.7):
    
    size = X.shape[1]
    dim = X.shape[0]
    dim_y = Y.shape[0]
    
    index = rnd.sample(range(size),size)
    
    mid = int(size*train_percentage)
    
    
    X_train = np.zeros((dim,mid))
    Y_train = np.zeros((dim_y,mid))
    
    X_test = np.zeros((dim,size-mid))
    Y_test = np.zeros((dim_y,size-mid))
    
    flag = 0
    
    for j in range(mid):
    
        a = np.reshape(X[:,index[flag]],(dim))
        b = np.reshape(Y[:,index[flag]],(dim_y))
        X_train[:,j] = a 
        Y_train[:,j] = b
        flag+=1
        
    for j in range(size-mid):
    
        a = np.reshape(X[:,index[flag]],(dim))
        b = np.reshape(Y[:,index[flag]],(dim_y))
        X_test[:,j] = a 
        Y_test[:,j] = b
        flag+=1
    

    return X_train,Y_train,X_test,Y_test

    
def normalizeImg(a):
    
    a = 2*a-1
    
    return a

def estimate(Net,sess,data,var,mean):
    
    
    value =Net.predict(data,sess)*var+mean
    
    return value


def evaluate_model(Net,sess,X_test,Y_test,acc):

    predict = estimate(Net,sess,X_test,1,0)
    
    error = np.abs(np.subtract(predict,Y_test)/Y_test).tolist()[0]
    
    right = 0
    wrong = 0
    
    for result in error:
        
        if result <= acc:
            right+=1
        else:
            wrong+=1
    
    total = right + wrong
    
    return [right/total, wrong/total]


def accuracy(Net,sess,data,targets,thresh = 0.5 ):
    
    depth = targets.shape[1]

    A = Net.predict(data,sess, output = 'class')
    A=np.argmax(A,axis=0).astype(int)
    
    A =np.reshape(A,(1,depth))
    
    A = one_hot_matrix(A)
    B = np.abs(targets - A)
    acc = 1 - np.sum(B)/2/depth
    print("Accuracy: %f\n" %  acc)
        
    
    
        