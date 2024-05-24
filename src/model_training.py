# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:14:07 2024

@author: vijayendra.dwari
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['sales'])
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

if __name__ == "__main__":
    model = train_model('../data/processed_data.csv')
