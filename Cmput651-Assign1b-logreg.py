# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:18:35 2019

@author: zehra
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import expit

X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(open("data.pkl", "rb"))
id2word, word2id = pickle.load( open("dicts.pkl", "rb") )

y_train = np.float64(np.expand_dims( np.array(y_train), axis=1 ))
y_val = np.float64(np.expand_dims( np.array(y_val), axis=1 ))
y_test = np.float64(np.expand_dims( np.array(y_test), axis=1 ))


learning_rate = 0.1#0.1 #(Total loss = average of per sample loss in the batch)
#Learning rate decay: None (fixed learning rate)
batch_size = 20
#Regularization: None
num_epochs = 300
num_features = np.shape(X_train)[1]
num_samples_train = np.shape(X_train)[0]


#Helper functions:
def sigmoid(x):
    return expit(x) # 1 / (1 + np.exp(-x)) #
    
def LRcost(t, pred):
    pred[ pred== 0.0 ] = 10**-10 #Add epsilon to all zero values, to avoid numerical underflow
    cost_per_sample = -t*np.log(pred) - (1-t)*np.log(1-pred)
    avg_cost = np.mean(cost_per_sample)
    return avg_cost  

def LRgradient_batch(X, y, pred):
    m = X.shape[0]
    grad = np.dot( X.T, (pred-y) ) #X.T*(prediction - target) #Dimension: 2000x1
    grad = (1/m) * grad #Divide by number of samples
    return grad

#Initializations:
train_cost_history = np.zeros(num_epochs)
val_cost_history = np.zeros(num_epochs)
train_accuracy = np.zeros(num_epochs)
val_accuracy = np.zeros(num_epochs)
test_accuracy = np.zeros(1)
theta_history = np.zeros((num_epochs,num_features))
theta_0_history = np.zeros(num_epochs)

#Parameter initialization: Uniform[-0.5, 0.5]
theta = np.random.uniform(low=-0.5, high=0.5, size=(num_features,1)) # theta: 2000 x 1
theta_0 = np.random.uniform(low=-0.5, high=0.5) #theta_0: scalar

#Training Loop: For epochs = 1, .., 300:
for epoch in range(num_epochs): 
    J = 0.0   #Logistic Regression Cost J (scalar)
    gradJ = np.zeros(theta.shape) #gradient of theta: 2000 x 1
    gradJ_0 = 0.0 #gradient of theta_0: scalar

    #Mini-batch gradient computation and theta update loop:
    for i in range(0, num_samples_train, batch_size): #For batches of the training set:
        X_i = X_train[i:i+batch_size] #X_i: 20x2000
        y_i = y_train[i:i+batch_size] #y_i: 20x1
        
        z_i = np.dot(X_i, theta) + theta_0 #z_i: 20x1
        pred_i = sigmoid(z_i) #pred_i: 20x1
        J += LRcost(y_i, pred_i) #Compute logistic regression cost for current batch 
        #Compute gradients:
        gradJ = LRgradient_batch(X_i, y_i, pred_i) 
        gradJ_0 = np.sum(pred_i-y_i)
        #Update the parameters:
        theta = theta - learning_rate*gradJ #theta: 2000x1
        theta_0 = theta_0 - learning_rate*gradJ_0 #theta_0: scalar
    #End mini-batch gradient / theta update loop
    
    
    #Predict on training set:
    z_train = np.dot(X_train, theta) + theta_0 #z_train: 20,000 x 1
    pred_train = sigmoid(z_train) #pred_train: 20,000 x 1
    pred_train_class = np.zeros(pred_train.shape)
    pred_train_class [ pred_train > 0.5 ] = 1.0 #pred_train_class: 20,000 x 1
    train_cost_history[epoch] = LRcost(y_train, pred_train)
    train_accuracy[epoch] = np.sum(y_train==pred_train_class)/len(y_train) 
    print("Epoch: "+str(epoch)+" | Accuracy on training set: "+str(train_accuracy[epoch]))
    
    #Predict on validation set:
    z_val = np.dot(X_val, theta) + theta_0 #z_val: 5,000 x 1
    pred_val = sigmoid(z_val) #pred_val: 5,000 x 1
    val_cost_history[epoch] = LRcost(y_val, pred_val)
    theta_history[epoch,] = np.squeeze(theta)
    theta_0_history[epoch] = theta_0
    pred_val_class = np.zeros(pred_val.shape)
    pred_val_class [ pred_val > 0.5 ] = 1.0 #pred_val_class: 5,000 x 1
    val_accuracy[epoch] = np.sum(y_val==pred_val_class)/len(y_val) 
    print("Epoch: "+str(epoch)+" | Accuracy on validation set: "+str(val_accuracy[epoch]))
#End Training Loop

#Choose the best validation model:
best_val_accuracy = np.amax(val_accuracy)
best_epoch_id = np.argmax(val_accuracy)
best_theta = theta_history [best_epoch_id,]
best_theta = np.expand_dims(best_theta, axis=1)
best_theta_0 = theta_0_history [best_epoch_id,]


#Predict on the test set:
z_test = np.dot(X_test, best_theta) + best_theta_0 #z_test: 25,000 x 1
pred_test = sigmoid(z_test) #pred_test: 25,000 x 1
test_cost = LRcost(y_test, pred_test)
print("Logistic Regression cost on the Test set: "+str(test_cost))
pred_test_class = np.zeros(pred_test.shape)
pred_test_class [ pred_test > 0.5 ] = 1.0 #pred_test_class: 25,000 x 1
test_accuracy = np.sum(y_test==pred_test_class)/len(y_test) 
print("LR test accuracy: "+str(test_accuracy))
    
#Plot Train/Val Accuracy:
plt.plot(train_accuracy)
plt.plot(val_accuracy)
plt.title('Model Accuracy (Learning Rate: '+str(learning_rate)+')')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
#axes = plt.gca()
#axes.set_ylim([0.5,1.0])
plt.show()

#Plot Train/Val Cost Function:
plt.figure()
plt.plot(train_cost_history)
plt.plot(val_cost_history)
plt.title('Logistic Regression Cost Function (Learning Rate: '+str(learning_rate)+')')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
    
 
