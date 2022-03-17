# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:46:11 2019

@author: ohadz
"""
###split the model to 80 and 20% of the data. 

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = \
train_test_split(dataset_Calibration_10min_NA_features, dataset_Calibration_10min_NA_label,\
                 test_size = 0.20, )
from sklearn.ensemble import RandomForestRegressor


### defines the model and it's hyper parameters
#  model with 1500 decision trees 
trees=1500 #roughly the optimum need to recheck for all sensors
leaves=10 # the minimum number of leaves per a split

rf = RandomForestRegressor(n_estimators = trees, n_jobs =-1, \
                           min_samples_leaf=leaves,criterion="mae",max_features="auto") 
# n_jobs means the CPU core used. -1 is all cores :)

# A smaller leaf makes the model more prone to capturing noise in train data.
# max_depth represents the depth of each tree in the forest. model overfits for large depth values

# Train the model on training data
rf.fit(train_features, train_labels);

# check some statistics of the model. 
from mlxtend.evaluate import feature_importance_permutation

def Spearman_func(X,Y): #spearman to use in the feature imporatnce checks
    tmp=st.spearmanr(X,Y)
    return tmp[0]

imp_vals, temp = feature_importance_permutation(predict_method=rf.predict, X=np.array(test_features), y=np.array(test_labels), metric=Spearman_func,num_rounds=10, seed=10)

#finds the linear regression for the extrapolation 
from sklearn.linear_model import LinearRegression
feature_max=max(zip(imp_vals,dataset_Calibration_10min_NA_features_list))[1]
dataset_linear=np.array([train_features[feature_max]])
dataset_linear=np.transpose(dataset_linear)
reg_lin=LinearRegression().fit(dataset_linear, np.array(train_labels))
