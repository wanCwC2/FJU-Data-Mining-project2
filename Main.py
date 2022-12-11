import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/project2_train.csv')
test = pd.read_csv('data/project2_test.csv')

########## Modify data ##########

#Age too small/big to use
#df = df.drop[df[(df['Age']<18) | (df['Age']>80)].index]
#TypeError: 'method' object is not subscriptable
#df[(df['Age']<18) | (df['Age']>80)].index: [119, 302, 323, 579, 595, 795, 877, 910]
df = df.drop([119, 302, 323, 579, 595, 795, 877, 910])

#Gender male/female but some strange
other  = ['A little about you', 'p', 'Nah', 'Enby', 'Trans-female','something kinda male?','queer/she/they','non-binary','All','fluid', 'Genderqueer','Androgyne', 'Agender','Guy (-ish) ^_^', 'male leaning androgynous','Trans woman','Neuter', 'Female (trans)','queer','ostensibly male unsure what that really means','trans']
male   = ['male', 'Male','M', 'm', 'Male-ish', 'maile','Cis Male','Mal', 'Male (CIS)','Make','Male ', 'Man', 'msle','cis male', 'Cis Man','Malr','Mail']
female = ['Female', 'female','Cis Female', 'F','f','Femake', 'woman','Female ','cis-female/femme','Female (cis)','femail','Woman','female']
df['Gender']= df['Gender'].replace(other, 0.5)
df['Gender']= df['Gender'].replace(male, 1)
df['Gender']= df['Gender'].replace(female, 0)
test['Gender']= test['Gender'].replace(other, 0.5)
test['Gender']= test['Gender'].replace(male, 1)
test['Gender']= test['Gender'].replace(female, 0)

#The Other data
for col in df.columns:
    df[col] = df[col].replace({'Yes': 1, 'No': 0, "Don't know": 0, 'Not sure': 0, 'Maybe':0.5, 'Some of them': 0.5})
    df[col] = df[col].replace({'Often': 1, 'Sometimes': 0.67, 'Rarely': 0.33, 'Never': 0})
    df[col] = df[col].replace({'More than 1000': 1, '500-1000': 0.8, '100-500': 0.6, '26-100': 0.4, '6-25': 0.2, '1-5': 0})
    df[col] = df[col].replace({'Somewhat easy': 0.25, "Don't know": 0.5, 'Somewhat difficult': 0.75, 'Very difficult': 1, 'Very easy': 0})
    df[col] = df[col].fillna(0)

for col in test.columns:
    test[col] = test[col].replace({'Yes': 1, 'No': 0, "Don't know": 0, 'Not sure': 0, 'Maybe':0.5, 'Some of them': 0.5})
    test[col] = test[col].replace({'Often': 1, 'Sometimes': 0.67, 'Rarely': 0.33, 'Never': 0})
    test[col] = test[col].replace({'More than 1000': 1, '500-1000': 0.8, '100-500': 0.6, '26-100': 0.4, '6-25': 0.2, '1-5': 0})
    test[col] = test[col].replace({'Somewhat easy': 0.25, "Don't know": 0.5, 'Somewhat difficult': 0.75, 'Very difficult': 1, 'Very easy': 0})
    test[col] = test[col].fillna(0)