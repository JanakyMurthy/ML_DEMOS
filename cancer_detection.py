#!/usr/bin/env python

''' cancer_detection.py - A simple classifier for classifying
    breast cancer as benign or malignant
'''

# imports...
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics

if __name__ == '__main__':

    # the dataset; Alternatively you can download the dataset and feed in the appropriate URL here
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"


    # Feature Selection
    names = ['id','clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
             'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
             'bland_chromatin','normal_nucleoli', 'mitosis', 'class']
    
    # Load the dataset
    dataset = pd.read_csv(data_url, names = names)

    # Understanding the data... Uncomment the lines with '##'
    # if you want to know more about data.
    
    ##print(dataset.head()) # first 5 data points
    ##print(dataset.axes)  # column names
    ##print(dataset.shape)
    ##print(dataset.loc[0]) # 0-th data point
    ##print(dataset.describe()) # some stats about the data

    # Let's do some dataset visualizations
    
    # Plot histograms for each variable
    ##dataset.hist(figsize = (10,20))
    ##plt.show()

    # create scatter plot matrix
    ##scatter_matrix(df, figsize = (18,18))
    ##plt.show()

    # Done with understanding the data! Let's get back to work

    # Some more data preprocessing.
    dataset.replace('?',-99999,inplace =  True) # fill out missing values
    dataset.drop(['id'], 1, inplace = True) # drop id value 

    # Split data set into X and Y
    X = np.array(dataset.drop(['class'],1))
    y = np.array(dataset['class'])
   
    # Set aside some data for testing
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)

    # Create a SVM classifier
    my_classifier = svm.SVC(gamma = 'auto')

    # Train it
    my_classifier.fit(X_train,y_train)

    # Evaluate it's performance on test data
    y_pred_test = my_classifier.predict(X_test)
    y_pred_train = my_classifier.predict(X_train)
    test_accuracy = metrics.accuracy_score(y_test,y_pred_test)
    train_accuracy = metrics.accuracy_score(y_train, y_pred_train)
    
    # Output performance
    print("Test accuracy: ", test_accuracy*100, "percent Training accuracy = ", train_accuracy*100, " percent") 
    
    
