import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/project2_train.csv')
test = pd.read_csv('data/project2_test.csv')

df['family_history'] = df['family_history'].replace({'Yes': 1, 'No': 0})