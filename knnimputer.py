# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 20:21:22 2020

@author: hemahemu
"""


import pandas as pd
dataset=pd.read_csv('G:/Major Project/mvdata.csv')
print(dataset.head())
print(dataset.columns)
print(dataset.info())
print(dataset.describe())
print(dataset.dtypes)

print(dataset.isnull().sum())
#print(dataset.astype(int))
#print(dataset.dtypes)
#from sklearn.model_selection import train_test_split
#X=dataset.iloc[:,:-1].values
#Y=dataset.iloc[:,1].values

#KNNImputer is to identify 'k' samples in the dataset that are similar or close in the space.
#Then we use these 'k' samples to estimate the value of the missing data points.
#Each sample's missing values are imputed using the mean value of the 'k'-neighbors found in the dataset
from sklearn.impute import KNNImputer
#imputer=KNNImputer(n_neighbors=2,weights="uniform")
imputer=KNNImputer()

x=imputer.fit_transform(dataset)
print(x)
