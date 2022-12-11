import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def standardization(data):
    sc = StandardScaler()   
    data = sc.fit_transform(data)
    return data

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

X_df = standardization(df.iloc[:,0:22])
y_df = df['treatment'].astype('int')
test = standardization(test)

X_train , X_test , y_train , y_test = train_test_split(X_df ,y_df , test_size=0.3 , random_state=408570344)

#XGBoost
from xgboost import XGBClassifier
'''
params = { 'max_depth': range (2, 15, 3),
           'learning_rate': [0.01, 0.1, 0.5, 1, 5, 10],
           'n_estimators': range(80, 500, 50),
           'colsample_bytree': [0.5, 1, 3, 6, 10],
#           'min_child_weigh': range(1, 9, 1),
           'subsample': [0.5, 0.7, 0.9, 1.5, 2]}

from sklearn.model_selection import GridSearchCV
model = XGBClassifier()
clf = GridSearchCV(estimator = model,
                   param_grid = params,
                   scoring = 'neg_log_loss')
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
# Best parameters: {'colsample_bytree': 1, 'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 380, 'subsample': 0.9}
print(clf.best_estimator_)
# XGBClassifier(colsample_bytree=1, learning_rate=0.01, max_depth=2, n_estimators=380, subsample=0.9)
'''
#model = clf.best_estimator_
model = XGBClassifier(colsample_bytree=1, learning_rate=0.01, max_depth=2, n_estimators=380, subsample=0.9)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred))) #0.83

#Output data
df_submit = pd.DataFrame([], columns=['Id', 'Treatment'])
df_submit['Id'] = [f'{i:03d}' for i in range(len(test))]
df_submit['Treatment'] = model.predict(test)

df_submit.to_csv('data/predict.csv', index=None)
