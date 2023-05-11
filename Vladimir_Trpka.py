# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:40:15 2021

@author: Vladimir Trpka IN41-2018


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import datasets

cengdu = pd.read_csv("ChengduPM20100101_20151231.csv")


#%%

print(cengdu.info())

#%%
print(cengdu.shape)
shape = cengdu.shape

#%%

tipovi = cengdu.dtypes

#%%

pet_prvih = cengdu.head()

""" Dataframe ima 52484 uzoraka i 17 obeležja.
    Kategorička obeležja su : year, month, day, hour, season, cbwd.
    Numerička obeležja su : No ,PM_Caotangsi, PM_Shahepu, PM_US Post, DEWP, 
                            TEMP, HUMI, PRES, Iws, precipitation, Iprec.
    Jedan uzorak predstavlja jedan sat u toku jednog dana u godini sa zabeleženim podacima o klimatskim uslovima i pm česticama.
"""

#%%

opis = cengdu.describe()

#%%

total = cengdu.isnull().sum().sort_values(ascending=False) 
percent = 100*(cengdu.isnull().sum()/len(cengdu)).sort_values(ascending=False)

"""
Nedostajući podaci se javljaju u kolonama : PM_Caotangsi(28164), PM_Shahepu(27990), PM_US Post(23684), precipitation(2955),
                                            Iprec(2955), HUMI(535), Iws(533), DEWP(529), TEMP(527), cbwd(521), PRES(521)
Nelogicnosti i nevalidne vrednosti ???
"""
#%%
"""Zamena vrenosti numeričkim vrednostima"""

print('cbwd: \n', cengdu['cbwd'].unique())

cengdu.loc[cengdu['cbwd']=='cv','cbwd']=0
cengdu.loc[cengdu['cbwd']=='NE','cbwd']=1
cengdu.loc[cengdu['cbwd']=='SW','cbwd']=2
cengdu.loc[cengdu['cbwd']=='NW','cbwd']=3
cengdu.loc[cengdu['cbwd']=='SE','cbwd']=4

#%%
"""Izbacivanje kolona"""

cengdu.drop(['PM_Caotangsi', 'PM_Shahepu', 'No' ], inplace= True, axis = 1)

"""Izbacujemo kolone PM_Caotangsi(iz zadatka), PM_Shahepu(iz zadatak), No(Redni brojevi nisu relevantni za obradu)""" 

#%%
"""Koliko ima nedostajućih vrednosti"""
print(cengdu.isnull().sum())

#%%
"""Izlistavanje nedostajućih vrednosti po zadatom kriterijumu.
Kolone Iprec i precipitation imaju imaju 2955 nedostajućih vrednosti, nedostajući podaci će biti
izbačeni jer vrednosti nije lako dopuniti nekom metodom. Isto tako i za HUMI I Iws"""
Iprec = cengdu[cengdu['Iprec'].isna()]
Humi = cengdu[cengdu['HUMI'].isna()]
Iws = cengdu[cengdu['Iws'].isna()]

#%%
"""Izbacivanje uzoraka koji sadrže nosstajuće vrenosti"""
cengdu.dropna(subset= ['PM_US Post','Iprec', 'precipitation','Iws','HUMI'], inplace=True, axis=0)
cengdu = cengdu.reset_index()
cengdu.drop(['index'], inplace=True, axis=1)

#%%
"""Provera da li ima još nedostajućih vrednosti"""
print(cengdu.isnull().sum())

#%%
"""Dinamički opseg"""
for i in cengdu.columns:
    a = cengdu[i].max() - cengdu[i].min()
    print("Dinamički opseg:",i,"je",a)

#%%

opis1 = cengdu.describe()
    
#%%

from scipy.stats import kurtosis
from scipy.stats import skew

##Pomerena u jednu stranu, nema simetricnost kao normalna
print('koef.asimetrije:  %.2f' % skew(cengdu.loc[:,'PM_US Post']))
print('koef.spljoštenosti:  %.2f' % kurtosis(cengdu.loc[:,'PM_US Post']))
##Raspodela ispod modelovane normalne raspodele

#%%


from scipy.stats import norm
cengdu_raspodela = cengdu.loc[:,'PM_US Post']
sns.distplot(cengdu_raspodela, fit=norm)
plt.xlabel('PM_US Post')
plt.ylabel('Verovatnoća')


#%%
"""Korelacija koeficijenti"""
corr = cengdu[["PM_US Post","DEWP","HUMI","PRES","TEMP","Iws","precipitation","Iprec"]].corr()
f = plt.figure(figsize=(12, 9))
sns.heatmap(corr.abs(), annot=True);

#%%
##print(cengdu[cengdu['PM_US Post']>450].count())
"""Iscrtavanje kombinacije obeležja kod kojih je primećena najveća korelacija"""
cengdu.plot.scatter(x='TEMP', y='DEWP', c="blue")

cengdu.plot.scatter(x='TEMP', y='PRES', c="black")

#%%
"""Temperatura"""
fig = sns.boxplot(x='year', y="TEMP", data=cengdu)
fig.axis(ymin=-2, ymax=40);

#%%
"""Međusobna kombinacija parova obeležja"""
cengdu1 = cengdu.drop(['year', 'month', 'day', 'hour', 'season'], axis = 1)

sns.set()
sns.pairplot(cengdu1, height = 2.5)
plt.show();

#%%
"""Prikaz zastupljenosti PM cestica"""
plt.boxplot([cengdu.loc[:, 'PM_US Post']]) 
plt.grid()
#%%
"""Prikaz zastupljenosti PM cestica tokom meseca u godini"""
##januar = cengdu.loc[cengdu['month']==1, 'HUMI']
jedinica = cengdu
mesec = jedinica.set_index('month')
mesec.head()
plt.figure()
f = plt.figure(figsize=(12, 9))
plt.boxplot([mesec.loc[1, 'PM_US Post'], mesec.loc[2, 'PM_US Post'], mesec.loc[3, 'PM_US Post'], mesec.loc[4, 'PM_US Post'], mesec.loc[5, 'PM_US Post'], mesec.loc[6, 'PM_US Post'], mesec.loc[7, 'PM_US Post'], mesec.loc[8, 'PM_US Post'], mesec.loc[9, 'PM_US Post'], mesec.loc[10,'PM_US Post'], mesec.loc[11, 'PM_US Post'], mesec.loc[12, 'PM_US Post']]) 

plt.xticks([1, 2, 3, 4 ,5, 6, 7, 8, 9, 10, 11, 12], ['Januar', 'Februar', 'Mart', 'April', 'Maj', 'Jun', 'Jul', 'Avgust', 'Septembar', 'Oktobar', 'Novembar', 'Decembar'])
plt.grid()

#%%
"""Prikaz zastupljenosti PM cestica kada je kisovito i kada je suvo vreme"""
kisa = cengdu.loc[cengdu['precipitation']!=0,:]
kisa_pm = kisa.loc[:,'PM_US Post'].tolist()
plt.boxplot([kisa.loc[:, 'PM_US Post']]) 
plt.grid()
suvo = cengdu.loc[cengdu['precipitation']==0 ,:]
suvo_pm = suvo.loc[:,'PM_US Post'].tolist()
plt.boxplot([suvo.loc[:, 'PM_US Post']]) 
plt.grid()

kisaopis = kisa.describe()
suvoopis = suvo.describe()


data = [kisa_pm, suvo_pm]

fig, ax = plt.subplots()
ax.boxplot(data)
plt.xticks([1, 2],["Kišovito", "Suvo"])
plt.grid()
plt.show()

#%%
"""Prikaz zastupljenosti PM cestica tokom godine"""
prolece = cengdu.loc[cengdu['season'].isin([1])]
prolece_pm = prolece.loc[:, 'PM_US Post']
leto = cengdu.loc[cengdu['season'].isin([2])]
leto_pm = leto.loc[:, 'PM_US Post']
jesen = cengdu.loc[cengdu['season'].isin([3])]
jesen_pm = jesen.loc[:, 'PM_US Post']
zima = cengdu.loc[cengdu['season'].isin([4])]
zima_pm = zima.loc[:, 'PM_US Post']

data1 = [prolece_pm, leto_pm, jesen_pm, zima_pm]

fig, ax = plt.subplots()
ax.boxplot(data1)
plt.xticks([1, 2, 3, 4],["Proleće", "Leto", "Jesen", "Zima"])
plt.grid()
plt.show()

#%%
"""Prikaz zastupljenosti PM cestica tokom dana"""
jutro = cengdu.loc[cengdu['hour'].isin([6,7,8,9,10,11])]
jutro_pm = jutro.loc[:, 'PM_US Post']
dan = cengdu.loc[cengdu['hour'].isin([12,13,14,15,16,17])]
dan_pm = dan.loc[:, 'PM_US Post']
vece = cengdu.loc[cengdu['hour'].isin([18,19,20,21,22,23])]
vece_pm = vece.loc[:, 'PM_US Post']
noc = cengdu.loc[cengdu['hour'].isin([24,1,2,3,4,5])]
noc_pm = noc.loc[:, 'PM_US Post']

data2 = [jutro_pm, dan_pm, vece_pm, noc_pm]

fig, ax = plt.subplots()
ax.boxplot(data2)
plt.xticks([1, 2, 3, 4],["Jutro", "Dan", "Veče", "Noć"])
plt.grid()
plt.show()


#%%
##Regresija 


x = cengdu.drop(['PM_US Post'], axis = 1)
y = cengdu['PM_US Post']

def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) 
    mae = mean_absolute_error(y_test, y_predicted) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

    # printing values
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

first_regression_model = LinearRegression(fit_intercept=True)

first_regression_model.fit(x_train, y_train)


y_predicted = first_regression_model.predict(x_test)


model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)


import statsmodels.api as sm
X = sm.add_constant(x_train)

model = sm.OLS(y_train, X.astype('float')).fit()
model.summary()
#%%

scaler = StandardScaler()
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
x_train_std = pd.DataFrame(x_train_std)
x_test_std = pd.DataFrame(x_test_std)
x_train_std.columns = list(x.columns)
x_test_std.columns = list(x.columns)
x_train_std.head()

regression_model_std = LinearRegression()


regression_model_std.fit(x_train_std, y_train)


y_predicted = regression_model_std.predict(x_test_std)


model_evaluation(y_test, y_predicted, x_train_std.shape[0], x_train_std.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_std.coef_)),regression_model_std.coef_)
plt.show()
print("koeficijenti: ", regression_model_std.coef_)

#%%

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

print(poly.get_feature_names())

regression_model_inter = LinearRegression()


regression_model_inter.fit(x_inter_train, y_train)


y_predicted = regression_model_inter.predict(x_inter_test)


model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter.coef_)),regression_model_inter.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter.coef_)

#%%

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_inter_train = poly.fit_transform(x_train_std)
x_inter_test = poly.transform(x_test_std)

regression_model_degree = LinearRegression()


regression_model_degree.fit(x_inter_train, y_train)


y_predicted = regression_model_degree.predict(x_inter_test)


model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_degree.coef_)),regression_model_degree.coef_)
plt.show()
print("koeficijenti: ", regression_model_degree.coef_)

#%%


ridge_model = Ridge(alpha=5)


ridge_model.fit(x_inter_train, y_train)


y_predicted = ridge_model.predict(x_inter_test)


model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])



plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

#%%


lasso_model = Lasso(alpha=0.01)


lasso_model.fit(x_inter_train, y_train)


y_predicted = lasso_model.predict(x_inter_test)


model_evaluation(y_test, y_predicted, x_inter_train.shape[0], x_inter_train.shape[1])



plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

plt.figure(figsize=(10,5))
plt.plot(regression_model_degree.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'linear',zorder=7) # zorder for ordering the markers
plt.plot(ridge_model.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Ridge') # alpha here is for transparency
plt.plot(lasso_model.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Lasso')
plt.xlabel('Coefficient Index',fontsize=16)
plt.ylabel('Coefficient Magnitude',fontsize=16)
plt.legend(fontsize=13,loc='best')
plt.show()
