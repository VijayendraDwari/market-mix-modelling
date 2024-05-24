# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:18:49 2024

@author: vijayendra.dwari
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from model_training import train_model

def evaluate_model(model, data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=['sales'])
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    return rmse, r2

if __name__ == "__main__":
    model = train_model('../data/processed_data.csv')
    rmse, r2 = evaluate_model(model, '../data/processed_data.csv')
    print(f'RMSE: {rmse}')
    print(f'R-squared: {r2}')
