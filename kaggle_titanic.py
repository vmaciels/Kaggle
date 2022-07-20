import sys

import pandas as pd
import numpy as np
import scipy
import random
import math
import os
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz

#importar os dados
os.chdir("/home/megamente/Downloads")
train_data="train.csv"

#cria um dataframe, e limpa os dados
data=pd.read_csv(train_data)
data.describe()
data=data.drop(['Fare','Cabin','Embarked','Ticket'],axis=1)
data=data.dropna(axis=0)

#substitui as variáveis parch e sibsp por algo que parece mais útil
tem_filho=[]
tem_pai=[]
for i in range(len(data)):
    if data.iloc[i].Age<15:
        tem_filho.append(0)
        tem_pai.append(data.iloc[i].Parch)
    else:
        tem_filho.append(data.iloc[i].Parch)
        tem_pai.append(0)
data['tem_filho']=tem_filho
data['tem_pai']=tem_pai

#deixa sexo em float
sex_bin=[]
for i in range(len(data)):
    if data.iloc[i].Sex=='male':
        sex_bin.append(1)
    elif data.iloc[i].Sex=='female':
        sex_bin.append(0)
data['sex_bin']=sex_bin

#separa o target e os dados de treino
feats=['Age','Pclass','SibSp','sex_bin']
x=data[feats]

y=data.Survived

#separa coisa de validação e treina
train_x,val_x,train_y,val_y=train_test_split(x,y,random_state=0)

titanic_model=RandomForestClassifier(random_state=42,max_depth=3)
titanic_model.fit(train_x,train_y)

#mostra o erro pra parte de validação
val_predictions = titanic_model.predict(val_x)
print(mean_absolute_error(val_y, val_predictions))

#faz as mesmas coisas de antes pro dado de teste
test='test.csv'
test_data=pd.read_csv(test)
sex_bin=[]
for i in range(len(test_data)):
    if data.iloc[i].Sex=='male':
        sex_bin.append(1)
    elif data.iloc[i].Sex=='female':
        sex_bin.append(0)
test_data['sex_bin']=sex_bin
tem_filho=[]
tem_pai=[]
for i in range(len(test_data)):
    if data.iloc[i].Age<15:
        tem_filho.append(0)
        tem_pai.append(test_data.iloc[i].Parch)
    else:
        tem_filho.append(test_data.iloc[i].Parch)
        tem_pai.append(0)
test_data['tem_filho']=tem_filho
test_data['tem_pai']=tem_pai

#substitui as idades que estão em NaN por uma normal
ls=[]
for i in range(len(test_data)):
    if not math.isnan(test_data.iloc[i].Age):
        ls.append(test_data.iloc[i].Age)
mean_age=np.mean(ls)
std_age=np.std(ls)

age_novo=[]
for i in range(len(test_data)):
    if math.isnan(test_data.iloc[i].Age):
        age_novo.append(scipy.stats.skewnorm.rvs(a=1,loc=mean_age,scale=std_age))
    else:
        age_novo.append(test_data.iloc[i].Age)
test_data=test_data.assign(Age=age_novo)
        
valx=test_data[feats]
valx=valx.dropna(axis=0)

#prevê as coisas e cria o csv
preds=titanic_model.predict(valx)
previsao=pd.DataFrame()
previsao['PassengerId']=test_data.PassengerId
previsao['Survived']=preds

previsao.to_csv('kaggle_titanic.csv',index=False)

#plt.figure(figsize=(10,10))  # customize according to the size of your tree
#_ = tree.plot_tree(titanic_model, feature_names = x.columns,
#                   filled=True, fontsize=6, rounded = True)
#plt.show()