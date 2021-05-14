# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:58:59 2021

@author: ali_k
"""


from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt
import random
import numpy as np
from math import sqrt
 

figure_count = 1

def getrandomweight():
    return round(random.uniform(-0.1,0.1), 3)

def get_phs(x_t, m_h, s_h):
    result = []
    for i in range(len(m_h)):
        result.append(np.exp(-(x_t - m_h[i])**2 / (2 * s_h[i]**2)))
    return result

def get_max_distance(centers):
    distances = []
    for center in centers:
        for other in centers:
            if center != other:
                distances.append(abs(center - other))
    return max(distances)

def predict(x_t, neurons_mh, neurons_sh, weights):
    phs = get_phs(x_t, neurons_mh, neurons_sh)
                
    # Bias value
    phs.append(1)
    
    return np.dot(weights, phs), phs


hidden_layer = [2, 4, 50]
num_of_trial = 10
n_epoch = 1000
learning_rate = 0.1


df_train = pd.read_csv('d_reg_tra.txt', delim_whitespace=True, header=None)
df_validate = pd.read_csv('d_reg_val.txt', delim_whitespace=True, header=None)

# keep errors for each number of clusters here
training_errors_by_h = []
validation_errors_by_h = []

predicted_train = []

for num_of_neurons in hidden_layer:
    print('num_of_neurons: ', num_of_neurons)
    
    model = MiniBatchKMeans(n_clusters = num_of_neurons)
    df_train['cluster'] = model.fit_predict(df_train.iloc[:, [0]])
    
    neurons_mh = []
    neurons_sh = []
    weights = []
    
    # keep errors for each different initial weights here
    training_errors = []
    validation_errors = []
    
    for neuron_index in range(num_of_neurons):
        neurons_mh.append(model.cluster_centers_[neuron_index][0])
        neurons_sh.append(get_max_distance(model.cluster_centers_.reshape(-1)) / sqrt(len(model.cluster_centers_)))
    
    # try with 10 different initial weights
    for i in range(num_of_trial):
        print('num_of_trial: ', i)
        
        # initialize perceptron
        weights = []
        for j in range(num_of_neurons + 1):
            weights.append(getrandomweight())
        
        # keep errors for each epoch here
        sub_training_errors = []
        sub_validation_errors = []
        
        # perceptron training phase
        for epoch in range(n_epoch):
            
            # Training
            training_error = 0
            for row in df_train.values:
                predicted, phs = predict(row[0], neurons_mh, neurons_sh, weights)
                
                training_error += mean_squared_error([row[1]], [predicted], squared=False)
                
                for j in range(len(weights)):
                    delta = learning_rate * (row[1] - predicted) * phs[j]
                    weights[j] = weights[j] + delta
            
            sub_training_errors.append(training_error / len(df_train))
            
            actuals = []
            predictions = []
            
            # Validation
            validation_error = 0
            for row in df_validate.values:
                predicted, phs = predict(row[0], neurons_mh, neurons_sh, weights)
                
                validation_error += mean_squared_error([row[1]], [predicted], squared=False)
                
                actuals.append(row[1])
                predictions.append(predicted)
                
            sub_validation_errors.append(validation_error / len(df_validate))
            
        training_errors.append(sub_training_errors[-1])
        validation_errors.append(sub_validation_errors[-1])
        
    training_errors_by_h.append(sum(training_errors) / len(training_errors))
    validation_errors_by_h.append(sum(validation_errors) / len(validation_errors))
    
    # plot the p_h(RBF), weighted values w_h*p_h, and the overall output, together with the training data
    sorted_train_df = df_train.sort_values(by=[0])
    
    
    # overall output and actual values
    figure_count += 1
    plt.figure(figure_count)
    plt.title("Part 1 - H: " + str(num_of_neurons))
    plt.scatter(sorted_train_df[0], sorted_train_df[1], label='train actual')
    train_predicted = []
    train_phs = []
    for row in sorted_train_df.values:
        pred, phs = predict(row[0], neurons_mh, neurons_sh, weights)
        train_predicted.append(pred)
        train_phs.append(phs)
        
    plt.plot(sorted_train_df[0], train_predicted, label='train predicted')
    
    # p_h values
    for i in range(np.array(train_phs).shape[1]):
        plt.plot(sorted_train_df[0], np.array(train_phs)[:,i], label='p_h values')
    
    # weighted values w_h*p_h
    for i in range(np.array(train_phs).shape[1]):
        plt.plot(sorted_train_df[0], np.array(train_phs)[:,i] * weights[i], label='w_h*p_h values')
    
    plt.legend()

figure_count += 1
plt.figure(figure_count)
plt.title("Part 2")
plt.plot(hidden_layer, training_errors_by_h, label='training_error')
plt.plot(hidden_layer, validation_errors_by_h, label='validation_error')
plt.legend()

