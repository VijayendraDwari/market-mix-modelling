# -*- coding: utf-8 -*-
"""
Author @VijayendraDwari
"""
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Add your preprocessing steps here
    return df

if __name__ == "__main__":
    df = load_data('../data/market_mix_data.csv')
    df = preprocess_data(df)
    df.to_csv('../data/processed_data.csv', index=False)

