import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/project2_train.csv')
test = pd.read_csv('data/project2_test.csv')

print(data.columns)
'''
Index(['Age', 'Gender', 'self_employed', 'family_history', 'work_interfere',
       'no_employees', 'remote_work', 'tech_company', 'benefits',
       'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave',
       'mental_health_consequence', 'phys_health_consequence', 'coworkers',
       'supervisor', 'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence', 'treatment'],
      dtype='object')
'''

for col in data.columns:
  print('Unique values in {} :'.format(col), len(data[col].unique()))
'''
Unique values in Age : 52
Unique values in Gender : 44
Unique values in self_employed : 3
Unique values in family_history : 2
Unique values in work_interfere : 5
Unique values in no_employees : 6
Unique values in remote_work : 2
Unique values in tech_company : 2
Unique values in benefits : 3
Unique values in care_options : 3
Unique values in wellness_program : 3
Unique values in seek_help : 3
Unique values in anonymity : 3
Unique values in leave : 5
Unique values in mental_health_consequence : 3
Unique values in phys_health_consequence : 3
Unique values in coworkers : 3
Unique values in supervisor : 3
Unique values in mental_health_interview : 3
Unique values in phys_health_interview : 3
Unique values in mental_vs_physical : 3
Unique values in obs_consequence : 2
Unique values in treatment : 2
'''

for col in data.columns:
  print(data[col].unique())
'''
['Female' 'M' 'Male' 'female' 'male' 'm' 'maile' 'Trans-female'
 'Cis Female' 'F' 'something kinda male?' 'Cis Male' 'Woman' 'f' 'Mal'
 'Male (CIS)' 'queer/she/they' 'non-binary' 'woman' 'Make' 'Nah' 'All'
 'Enby' 'fluid' 'Genderqueer' 'Androgyne' 'cis-female/femme'
 'Guy (-ish) ^_^' 'male leaning androgynous' 'Male ' 'Trans woman' 'Man'
 'msle' 'Neuter' 'queer' 'Female (cis)' 'Mail' 'cis male'
 'A little about you' 'Malr' 'p' 'femail' 'Cis Man'
 'ostensibly male unsure what that really means']
[nan 'Yes' 'No']
['No' 'Yes']
['Often' 'Rarely' 'Never' 'Sometimes' nan]
['6-25' 'More than 1000' '26-100' '100-500' '1-5' '500-1000']
['No' 'Yes']
['Yes' 'No']
['Yes' "Don't know" 'No']
['Not sure' 'No' 'Yes']
['No' "Don't know" 'Yes']
['Yes' "Don't know" 'No']
['Yes' "Don't know" 'No']
['Somewhat easy' "Don't know" 'Somewhat difficult' 'Very difficult'
 'Very easy']
['No' 'Maybe' 'Yes']
['No' 'Yes' 'Maybe']
['Some of them' 'No' 'Yes']
['Yes' 'No' 'Some of them']
['No' 'Yes' 'Maybe']
['Maybe' 'No' 'Yes']
['Yes' "Don't know" 'No']
['No' 'Yes']
['Yes' 'No']
'''