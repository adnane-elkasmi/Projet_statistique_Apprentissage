#!/usr/bin/env python
# coding: utf-8

# ## Importer les bibliothèques

# In[2]:


import copy
import datetime
import random
from collections import Counter
from datetime import date

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import quandl
import seaborn as sns
import plotly.express as px

import yfinance as yf
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from pandas import read_csv
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR, SVC
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from yahoofinancials import YahooFinancials
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
import statsmodels.api as sm
from statsmodels.gam.generalized_additive_model import GLMGam
from nltk.corpus import treebank
from nltk.tag import hmm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import KernelPCA
from tensorflow.keras.utils import plot_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn import preprocessing
from sklearn import utils
from sklearn import linear_model
from sklearn.linear_model import TweedieRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.linear_model import ElasticNet
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.cm as cm
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import TheilSenRegressor
from matplotlib import pyplot
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.stats.stattools import durbin_watson
from tensorflow.keras.models import Sequential
from statsmodels.graphics.gofplots import qqplot
import warnings
warnings.filterwarnings("ignore")
import numpy as np 
import pylab 
import scipy.stats as stats

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures


# ### Importer la base de données de yahoo finance.

# In[3]:


def yahoo_df(_start_date='2013-01-01', _end_date='2021-11-30',_time_interval='daily'):
    yahoo_financials = YahooFinancials("BTC-USD")
    dict_data = yahoo_financials.get_historical_price_data(start_date = _start_date, end_date = _end_date, time_interval='daily')
    data = pd.DataFrame(dict_data['BTC-USD']['prices'])
    _bitcoin_df = data.drop(["date"],axis = 1).rename(columns={'formatted_date': 'date'}).set_index(["date"])
    return _bitcoin_df


# ### Importer des bases de données en rapport avec la blockchain et Importer la base de données finale.

# In[56]:


pd.options.display.max_rows = 6
data_with_na = pd.read_csv("data_with_na.csv")


# ## Traitement des données manquantes

# ### Premier traitement à l'aide l'interpolation.

# In[57]:


def how_much_missing_date(data):
    '''
    Compte le nombre de données manquantes dans un dataframe
    '''
    data_with_na = data.copy()
    na_step_1 = pd.DataFrame(data_with_na.isna().sum(),columns=['données manquantes'])
    pd.options.display.max_rows = None
    return na_step_1

how_much_missing_date(data_with_na)


# In[58]:


data_interpolate = data_with_na.interpolate()
how_much_missing_date(data_interpolate)


# ### Deuxième traitement à l'aide d'Iterative Imputer. 

# In[59]:


cov = data_interpolate.corr().style.background_gradient(cmap='coolwarm')
cov


# In[60]:


def imputation_by_iterativeimputer(df):
# Impute une base de données à en utilisant la méthode IterativeImputer
    _df = df # DataFrame
    _id =_df.index
    _col = df.columns
    imputation_mean = IterativeImputer(max_iter = 100, random_state=0 , initial_strategy = 'constant')
    imputation_mean.fit(_df)
    array_imputed_esp = imputation_mean.transform(_df)
    data_imputed_esp = pd.DataFrame(array_imputed_esp,columns = _col, index = _id )
    return data_imputed_esp


# In[61]:


data_iterative = imputation_by_iterativeimputer(data_interpolate.set_index(["date"])).drop(["total-bitcoins","utxo-count"],axis =1)


# In[62]:


how_much_missing_date(data_iterative)


# Il ne reste plus de valeurs manquantes.

# In[63]:


final_data = data_iterative


# ## Analyse descriptive des données 

# In[64]:


final_data.index = final_data.index.astype(dtype ='datetime64[ns]')
# Variable que l'on souhaite prédire.
final_data['Predict close'] = final_data['close'].shift(-10)


# In[65]:


pd.options.display.max_rows = 12
final_data


# ### Représenation graphique

# In[66]:


def show_graphic(data,x='date' ,y="close",title = "Prix du bitcoin", ylabel = "USD" ):
    final_data = data.copy()
    final_data.reset_index().plot(x=x,y = y,figsize=(15,7),grid=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.show()
show_graphic(final_data)


# ### Histogramme

# In[67]:


def histogram(data,feature, title = "Prix du bitcoin"):
    final_data = data.copy()
    final_data.hist(f"{feature}",figsize=(25,10))
    plt.title(title)
    plt.show()
histogram(final_data,"close")


# ### Coefficient de corrélation de Pearson

# In[68]:


def show_correlation_between(feature_1,data):
    final_data = data.copy()
    corr = final_data.corr().loc[[f"{feature_1}"],:].sort_values(by=f"{feature_1}",axis=1).drop(columns=[f"{feature_1}"])
    plt.figure(figsize = (35,5))
    sns.heatmap(corr,cmap='tab10',square=True,annot=True,cbar=False,annot_kws={"fontsize":16},linewidths=1, linecolor='black',)
    plt.title("Correlation avec les autres variables")
    plt.show()   


# In[69]:


show_correlation_between("Predict close",final_data)


# ### Q-Q-plot

# In[70]:


def QQ_plot(liste):
    ("")
    fig,axs = plt.subplots(1,len(liste),sharey=False,figsize=(7*len(liste),7))
    colors = plt.rcParams["axes.prop_cycle"]()
    for i,feat in enumerate(liste):
        x = sorted(list(final_data["close"]))
        y = sorted(list(final_data[feat]))

        axs[i].scatter(x,y,marker=".",color=next(colors)["color"],label=feat)

        line = mlines.Line2D([0, 1], [0, 1], color='black',linestyle=':',label='Normal Line')
        transform = axs[i].transAxes
        line.set_transform(transform)
        axs[i].add_line(line)
        axs[i].set_ylabel(feat)
        axs[i].grid()
        axs[i].set_xlabel("close")
        axs[i].legend()
    plt.savefig('qq_plot_1.png')
    plt.show()


# In[71]:


liste_1 = ['high','close','low','open']
QQ_plot(liste_1)


# In[72]:


liste_2 = ['market-price','cost-per-transaction','estimated-transaction-volume-usd','hash-rate', 'difficulty']
QQ_plot(liste_2)


# ### Marice de correlation

# In[73]:


final_data.corr().style.background_gradient(cmap='coolwarm')


# ## Feature Engineering

# ### Transformations

# In[74]:


def transformation(final_data,raw_feature):
    _final_data = final_data.copy()
    for feature in raw_feature:
        for j in [1,3,7,15,30,60,90]:
            _final_data[f"sma {feature} {j}"] = ta.sma(_final_data[f'{feature}'],j)
            _final_data[f"variance {feature} {j}"] = ta.variance(_final_data[f'{feature}'],j)
            _final_data[f"stdev {feature} {j}"] = ta.stdev(_final_data[f'{feature}'],j)
            _final_data[f"ema {feature} {j}"] = ta.ema(_final_data[f'{feature}'],j)
            _final_data[f"dema {feature} {j}"] = ta.dema(_final_data[f'{feature}'],j)
            _final_data[f"tema {feature} {j}"] = ta.tema(_final_data[f'{feature}'],j)
            _final_data[f"rsi {feature} {j}"] = ta.rsi(_final_data[f'{feature}'],j)
    return _final_data


# In[75]:


__final_data = transformation(final_data,final_data.columns.drop(['Predict close'])).fillna(method='bfill').reset_index().fillna(method='bfill').fillna(method='ffill')


# In[76]:


def indicator_from_ta_library(data):
    __final_data = data.copy()
    __final_data = add_all_ta_features(__final_data, open="Open", high='high', low="low", close="close", volume="volume", fillna=True)
    return __final_data


# In[77]:


data_ta = indicator_from_ta_library(__final_data)
data_ta


# ### Mise à l'échelle des données.

# In[78]:


def data_scaled(final_data):
    __final_data = final_data.copy()
    X = __final_data.drop(['date','Predict close'],axis=1)
    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X[X.columns])
    scaler = MinMaxScaler()
    X_scaled[X.columns] =  scaler.fit_transform(X_scaled[X.columns])
    return X_scaled


# In[79]:


pd.options.display.max_rows = 12
robust_data = data_scaled(data_ta)


# ### Sélection de variables.

# In[80]:


#on construit une fôrets de 1200 arbres et on retient 1700 variables pour chaqu'un des arbres.(cela va prendre 60 min !!)
y =final_data['Predict close'].interpolate()
rf = RandomForestRegressor(n_estimators=1200,max_features = 1700,criterion = "mae",n_jobs=-1,bootstrap=True,verbose=2,random_state=1)
rf.fit(robust_data,y)


# In[101]:


pd.options.display.max_rows = 6
feat_imp_df = pd.DataFrame(data = {"Feature Name": robust_data.columns,"Feature Importance":rf.feature_importances_})

feat_imp_df = feat_imp_df.sort_values("Feature Importance",ascending=False)[:]

fig,ax = plt.subplots(figsize=(14,12))
ax = sns.barplot(x = "Feature Importance",y = "Feature Name",
                data=feat_imp_df.sort_values("Feature Importance",ascending=False)[:4],palette="nipy_spectral")
plt.title('Feature Importances')

data_ready = robust_data[feat_imp_df["Feature Name"].values]
date = pd.date_range(start='2014-09-16', periods=len(data_ready), freq='D')
__data_ready = data_ready.assign(date =  date)
__data_ready = __data_ready.merge(pd.DataFrame(final_data['Predict close']), how ="left",on="date").drop(0)[:-8]


# In[84]:


# On selectionne les 4 variables les plus importantes.
__data_ready = __data_ready[["others_cr", "variance mempool-size 1", "variance mempool-size 30", "stdev mempool-size 1",'Predict close',"date"]][:-7]


# In[85]:


__data_ready


# # Cross validation

# ### Cross validation pour les série temporelles.

# In[86]:


train_window = 280
test_window = 46
train_splits = []
test_splits = []

for i in range(train_window, len(__data_ready),test_window):
    train_split = __data_ready[i-train_window:i]
    test_split = __data_ready[i:i+test_window]
    train_splits.append(train_split)
    test_splits.append(test_split)


# # Modelisation

# ### GLM

# In[35]:


import statsmodels.api as sm

glm_date_array = []
glm_y_test_array = []
glm_y_test_pred_array = []
glm_batch_id_array = []
glm_batch_id_array_result = []
glm_batch_mae_train_array = []
glm_batch_rmse_train_array = []
glm_batch_mae_test_array = []
glm_batch_rmse_test_array = []
glm_residuals = []

for i in range(len(train_splits)):
    
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1)
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1)
    
    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values
    

    Xtrain_split=sm.add_constant(Xtrain_split)
    Xtest_split=sm.add_constant(Xtest_split)
    glm_model = sm.GLM(ytrain_split, Xtrain_split, family=sm.families.Gaussian(link=sm.families.links.identity))
    glm_model = glm_model.fit()
    

    ytrain_pred = glm_model.predict(Xtrain_split)
    ytest_pred = glm_model.predict(Xtest_split)
    
    glm_residuals.extend(ytest_split-ytest_pred)
    glm_date_array.extend(test_splits[i]['date'])
    glm_y_test_array.extend(test_splits[i]['Predict close'])
    glm_y_test_pred_array.extend((ytest_pred.values.flatten()))
    glm_batch_id_array.extend([i]*len(test_splits[i]))
    
    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred)
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred,squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred)
    RMSE_test = mean_squared_error(ytest_split,ytest_pred,squared=False)
    
    glm_batch_id_array_result.append(i)
    glm_batch_mae_train_array.append(MAE_train)
    glm_batch_rmse_train_array.append(RMSE_train)
    glm_batch_mae_test_array.append(MAE_test)
    glm_batch_rmse_test_array.append(RMSE_test)


# In[36]:


glm_result_test_df = pd.DataFrame()
glm_result_test_df['paquet'] = glm_batch_id_array
glm_result_test_df['Date'] = glm_date_array
glm_result_test_df['y_test'] = glm_y_test_array
glm_result_test_df['y_test_pred'] = glm_y_test_pred_array
glm_y_test_array = glm_result_test_df['y_test']
glm_y_test_pred_array = glm_result_test_df['y_test_pred']

glm_result_metrics_df = pd.DataFrame()
glm_result_metrics_df['paquet'] = glm_batch_id_array_result
glm_result_metrics_df['MAE_train'] = glm_batch_mae_train_array
glm_result_metrics_df['RMSE_train'] = glm_batch_rmse_train_array
glm_result_metrics_df['MAE_test'] = glm_batch_mae_test_array
glm_result_metrics_df['RMSE_test'] = glm_batch_rmse_test_array

glm_result_metrics_df
pd.DataFrame(data = {"Résultats GLM" :glm_result_metrics_df.mean()}).drop(['paquet'],axis=0)


# In[37]:


plt.figure(figsize=(30,10))
plt.plot(glm_date_array,glm_result_test_df["y_test"],label='actual')
plt.plot(glm_date_array,glm_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[38]:


print(glm_model.summary())


# In[39]:


# Le score R²
sumofsquares = 0
sumofresiduals = 0
for i in range(len(glm_result_test_df["y_test"])) :
    sumofsquares += (glm_result_test_df["y_test"][i] - np.mean(glm_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (glm_result_test_df["y_test"][i] - glm_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[40]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(glm_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[41]:


measurements = glm_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[42]:


plt.figure(figsize=(10,7))
plt.scatter(glm_result_test_df["y_test"].values, glm_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[43]:


plt.figure(figsize=(30,7))
plt.plot(glm_date_array,glm_residuals)


# In[44]:


durbin_watson(np.array(glm_residuals))


# ## GAM

# In[45]:


from pygam import LogisticGAM, GAM

gam_date_array = []
gam_y_test_array = []
gam_y_test_pred_array = []
gam_batch_id_array = []
gam_batch_id_array_result = []
gam_batch_mae_train_array = []
gam_batch_rmse_train_array = []
gam_batch_mae_test_array = []
gam_batch_rmse_test_array = []
gam_residuals = []

for i in range(len(train_splits)):
    
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1)
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1)
    
    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values
    
    gam_model = GAM(distribution='normal',link='identity')
    gam_model.fit(Xtrain_split, ytrain_split)

    ytrain_pred = gam_model.predict(Xtrain_split)
    ytest_pred = gam_model.predict(Xtest_split)
    
    gam_residuals.extend(ytest_split-ytest_pred)
    gam_date_array.extend(test_splits[i]['date'])
    gam_y_test_array.extend(test_splits[i]['Predict close'])
    gam_y_test_pred_array.extend((ytest_pred.flatten()))
    gam_batch_id_array.extend([i]*len(test_splits[i]))
    
    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred)
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred,squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred)
    RMSE_test = mean_squared_error(ytest_split,ytest_pred,squared=False)
    
    gam_batch_id_array_result.append(i)
    gam_batch_mae_train_array.append(MAE_train)
    gam_batch_rmse_train_array.append(RMSE_train)
    gam_batch_mae_test_array.append(MAE_test)
    gam_batch_rmse_test_array.append(RMSE_test)


# In[55]:


gam_result_test_df = pd.DataFrame()
gam_result_test_df['paquet'] = gam_batch_id_array
gam_result_test_df['Date'] = gam_date_array
gam_result_test_df['y_test'] = gam_y_test_array
gam_result_test_df['y_test_pred'] = gam_y_test_pred_array
gam_y_test_array = gam_result_test_df['y_test']
gam_y_test_pred_array = gam_result_test_df['y_test_pred']

gam_result_metrics_df = pd.DataFrame()
gam_result_metrics_df['paquet'] = gam_batch_id_array_result
gam_result_metrics_df['MAE_train'] = gam_batch_mae_train_array
gam_result_metrics_df['RMSE_train'] = gam_batch_rmse_train_array
gam_result_metrics_df['MAE_test'] = gam_batch_mae_test_array
gam_result_metrics_df['RMSE_test'] = gam_batch_rmse_test_array

gam_result_metrics_df
pd.DataFrame(data = {"Résultats GAM" :gam_result_metrics_df.mean()}).drop(['paquet'],axis=0)


# In[47]:


plt.figure(figsize=(30,10))
plt.plot(gam_date_array,gam_result_test_df["y_test"],label='actual')
plt.plot(gam_date_array,gam_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[48]:


print(gam_model.summary())


# In[49]:


# Le score R²
sumofsquares = 0
sumofresiduals = 0
for i in range(len(gam_result_test_df["y_test"])) :
    sumofsquares += (gam_result_test_df["y_test"][i] - np.mean(gam_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (gam_result_test_df["y_test"][i] - gam_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[50]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(gam_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[51]:


measurements = gam_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[52]:


plt.figure(figsize=(10,7))
plt.scatter(gam_result_test_df["y_test"].values, gam_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[53]:


plt.figure(figsize=(30,7))
plt.plot(gam_date_array,gam_residuals)


# In[54]:


durbin_watson(np.array(gam_residuals))


# ## SVM (SVR)

# In[130]:


svr_date_array = []
svr_y_test_array = []
svr_y_test_pred_array = []
svr_batch_id_array = []
svr_batch_id_array_result = []
svr_batch_mae_train_array = []
svr_batch_rmse_train_array = []
svr_batch_mae_test_array = []
svr_batch_rmse_test_array = []
svr_residuals = []

for i in range(len(train_splits)):
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1).values
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1).values

    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values

    svr = SVR(C=10000,gamma='auto',kernel='linear')
    svr.fit(Xtrain_split, ytrain_split)

    ytrain_pred = svr.predict(Xtrain_split)
    ytest_pred = svr.predict(Xtest_split)

    svr_residuals.extend(ytest_split-ytest_pred)
    
    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred)
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred,squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred)
    RMSE_test = mean_squared_error(ytest_split,ytest_pred,squared=False)

    svr_date_array.extend(test_splits[i]['date'])
    svr_y_test_array.extend(test_splits[i]['Predict close'])
    svr_y_test_pred_array.extend((ytest_pred.flatten()))
    svr_batch_id_array.extend([i]*len(test_splits[i]))

    svr_batch_id_array_result.append(i)
    svr_batch_mae_train_array.append(MAE_train)
    svr_batch_rmse_train_array.append(RMSE_train)
    svr_batch_mae_test_array.append(MAE_test)
    svr_batch_rmse_test_array.append(RMSE_test)

svr_result_test_df = pd.DataFrame()
svr_result_test_df['batch_id'] = svr_batch_id_array
svr_result_test_df['Date'] = svr_date_array
svr_result_test_df['y_test'] = svr_y_test_array
svr_result_test_df['y_test_pred'] = svr_y_test_pred_array
svr_y_test_array = svr_result_test_df['y_test']
svr_y_test_pred_array = svr_result_test_df['y_test_pred']
svr_result_metrics_df = pd.DataFrame()
svr_result_metrics_df['batch_id'] = svr_batch_id_array_result
svr_result_metrics_df['MAE_train'] = svr_batch_mae_train_array
svr_result_metrics_df['RMSE_train'] = svr_batch_rmse_train_array
svr_result_metrics_df['MAE_test'] = svr_batch_mae_test_array
svr_result_metrics_df['RMSE_test'] = svr_batch_rmse_test_array


# In[131]:


pd.DataFrame(data = {"résultats SVR" :svr_result_metrics_df.mean()}).drop(['batch_id'])


# In[151]:


pd.DataFrame(svr_result_metrics_df.mean()).drop(['batch_id'])
plt.figure(figsize=(30,10))
plt.plot(svr_date_array,svr_result_test_df["y_test"],label='actual')
plt.plot(svr_date_array,svr_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[132]:


# Le score R²
sumofsquares = 0
sumofresiduals = 0
for i in range(len(svr_result_test_df["y_test"])) :
    sumofsquares += (svr_result_test_df["y_test"][i] - np.mean(svr_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (svr_result_test_df["y_test"][i] - svr_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[133]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(svr_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[134]:


measurements = svr_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[135]:


plt.figure(figsize=(10,7))
plt.scatter(svr_result_test_df["y_test"].values, svr_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[136]:


plt.figure(figsize=(30,7))
plt.plot(svr_date_array,svr_residuals)


# In[137]:


durbin_watson(np.array(svr_residuals))


# ## Regression Lasso

# #### GridsearchCV
# 
# GridSearchCV est une méthode d’optimisation qui va nous permet de tester une série de paramètres et de comparer les performances pour en déduire le meilleur paramétrage.
# 
# On utilise GridSearchCV afin de trouver le bon $\alpha \in [1,32]$.

# In[139]:


lasso_date_array = []
lasso_y_test_array = []
lasso_y_test_pred_array = []
lasso_batch_id_array = []
lasso_batch_id_array_result = []
lasso_batch_mae_train_array = []
lasso_batch_rmse_train_array = []
lasso_batch_mae_test_array = []
lasso_batch_rmse_test_array = []
lasso_residuals = []

for i in range(len(train_splits)):
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1)
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1)
    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values
    
    alphas = np.arange(1,32)
    model = Lasso()
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))
    lasso_model = grid.fit(Xtrain_split,ytrain_split)
    ytrain_pred = lasso_model.predict(Xtrain_split)
    ytest_pred = lasso_model.predict(Xtest_split)

    lasso_date_array.extend(test_splits[i]['date'])
    lasso_y_test_array.extend(test_splits[i]['Predict close'])
    lasso_y_test_pred_array.extend((ytest_pred.flatten()))
    lasso_batch_id_array.extend([i]*len(test_splits[i]))
    
    lasso_residuals.extend(ytest_split-ytest_pred)
    
    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred)
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred,squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred)
    RMSE_test = mean_squared_error(ytest_split,ytest_pred,squared=False)
    
    lasso_batch_id_array_result.append(i)
    lasso_batch_mae_train_array.append(MAE_train)
    lasso_batch_rmse_train_array.append(RMSE_train)
    lasso_batch_mae_test_array.append(MAE_test)
    lasso_batch_rmse_test_array.append(RMSE_test)


# In[142]:


lasso_result_test_df = pd.DataFrame()
lasso_result_test_df['paquet'] = lasso_batch_id_array
lasso_result_test_df['Date'] = lasso_date_array
lasso_result_test_df['y_test'] = lasso_y_test_array
lasso_result_test_df['y_test_pred'] = lasso_y_test_pred_array
lasso_y_test_array = lasso_result_test_df['y_test']
lasso_y_test_pred_array = lasso_result_test_df['y_test_pred']

lasso_result_metrics_df = pd.DataFrame()
lasso_result_metrics_df['paquet'] = lasso_batch_id_array_result
lasso_result_metrics_df['MAE_train'] = lasso_batch_mae_train_array
lasso_result_metrics_df['RMSE_train'] = lasso_batch_rmse_train_array
lasso_result_metrics_df['MAE_test'] = lasso_batch_mae_test_array
lasso_result_metrics_df['RMSE_test'] = lasso_batch_rmse_test_array

pd.DataFrame(data = {"Résultats Régression LASSO" : lasso_result_metrics_df.mean()}).drop(['paquet'],axis=0)


# In[148]:


plt.figure(figsize=(30,10))
plt.plot(lasso_date_array,lasso_result_test_df["y_test"],label='actual')
plt.plot(lasso_date_array,lasso_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[141]:


# Le score R²
sumofsquares = 0
sumofresiduals = 0
for i in range(len(lasso_result_test_df["y_test"])) :
    sumofsquares += (lasso_result_test_df["y_test"][i] - np.mean(lasso_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (lasso_result_test_df["y_test"][i] - lasso_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[143]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(lasso_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[144]:


measurements = lasso_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[145]:


plt.figure(figsize=(10,7))
plt.scatter(lasso_result_test_df["y_test"].values, lasso_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[146]:


plt.figure(figsize=(30,7))
plt.plot(lasso_date_array,lasso_residuals)


# In[147]:


durbin_watson(np.array(lasso_residuals))


# ## Regression ridge

# On utilise GridSearchCV afin de trouver le bon $\alpha \in [1,30]$.

# In[154]:


ridge_date_array = []
ridge_y_test_array = []
ridge_y_test_pred_array = []
ridge_batch_id_array = []
ridge_batch_id_array_result = []
ridge_batch_mae_train_array = []
ridge_batch_rmse_train_array = []
ridge_batch_mae_test_array = []
ridge_batch_rmse_test_array = []
ridge_residuals = []

for i in range(len(train_splits)):
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1)
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1)
    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values
    
    alphas = np.arange(1,30)
    model = Ridge()
    params_grid = { 
    'alpha': alphas,
    'normalize' : [True,False]}
    grid = GridSearchCV(estimator=model, param_grid=params_grid)
    ridge_model = grid.fit(Xtrain_split,ytrain_split)
    ytrain_pred = ridge_model.predict(Xtrain_split)
    ytest_pred = sgd_reg.predict(Xtest_split)

    ridge_date_array.extend(test_splits[i]['date'])
    ridge_y_test_array.extend(test_splits[i]['Predict close'])
    ridge_y_test_pred_array.extend((ytest_pred.flatten()))
    ridge_batch_id_array.extend([i]*len(test_splits[i]))
    
    ridge_residuals.extend(ytest_split-ytest_pred)

    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred)
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred,squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred)
    RMSE_test = mean_squared_error(ytest_split,ytest_pred,squared=False)
    
    ridge_batch_id_array_result.append(i)
    ridge_batch_mae_train_array.append(MAE_train)
    ridge_batch_rmse_train_array.append(RMSE_train)
    ridge_batch_mae_test_array.append(MAE_test)
    ridge_batch_rmse_test_array.append(RMSE_test)


# In[155]:


ridge_result_test_df = pd.DataFrame()
ridge_result_test_df['paquet'] = ridge_batch_id_array
ridge_result_test_df['Date'] = ridge_date_array
ridge_result_test_df['y_test'] = ridge_y_test_array
ridge_result_test_df['y_test_pred'] = ridge_y_test_pred_array
ridge_y_test_array = ridge_result_test_df['y_test']
ridge_y_test_pred_array = ridge_result_test_df['y_test_pred']

ridge_result_metrics_df = pd.DataFrame()
ridge_result_metrics_df['paquet'] = ridge_batch_id_array_result
ridge_result_metrics_df['MAE_train'] = ridge_batch_mae_train_array
ridge_result_metrics_df['RMSE_train'] = ridge_batch_rmse_train_array
ridge_result_metrics_df['MAE_test'] = ridge_batch_mae_test_array
ridge_result_metrics_df['RMSE_test'] = ridge_batch_rmse_test_array

pd.DataFrame(data = {"Résultats régression ridge" : ridge_result_metrics_df.mean()}).drop(['paquet'],axis=0)


# In[156]:


plt.figure(figsize=(30,10))
plt.plot(ridge_date_array,ridge_result_test_df["y_test"],label='actual')
plt.plot(ridge_date_array,ridge_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[157]:


# Le score R² 
sumofsquares = 0
sumofresiduals = 0
for i in range(len(ridge_result_test_df["y_test"])) :
    sumofsquares += (ridge_result_test_df["y_test"][i] - np.mean(ridge_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (ridge_result_test_df["y_test"][i] - ridge_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[158]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(ridge_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[159]:


measurements = ridge_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[160]:


plt.figure(figsize=(10,7))
plt.scatter(ridge_result_test_df["y_test"].values, ridge_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[161]:


plt.figure(figsize=(30,7))
plt.plot(ridge_date_array,ridge_residuals)


# In[162]:


durbin_watson(np.array(ridge_residuals))


# ## RNN LSTM 

# In[40]:


lstm_date_array = []
lstm_y_test_array = []
lstm_y_test_pred_array = []
lstm_batch_id_array = []
lstm_batch_id_array_result = []
lstm_batch_mae_train_array = []
lstm_batch_rmse_train_array = []
lstm_batch_mae_test_array = []
lstm_batch_rmse_test_array = []
lstm_residuals = []

for i in range(len(train_splits)):
    Xtrain_split = train_splits[i].drop(['Predict close','date'],axis=1)
    Xtest_split = test_splits[i].drop(['Predict close','date'],axis=1)
    ytrain_split = train_splits[i]['Predict close'].reset_index(drop=True).values
    ytest_split = test_splits[i]['Predict close'].reset_index(drop=True).values
    n_steps = 3
    n_features = 1
    Xtrain_split = np.array(Xtrain_split).reshape((Xtrain_split.shape[0], Xtrain_split.shape[1], n_features))
    model = Sequential()
    model.add(LSTM(units =50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse')
    model.fit(Xtrain_split, np.array(ytrain_split), batch_size = 32,epochs=200, verbose=0)
    ytrain_pred = model.predict(Xtrain_split,verbose=0)
    ytest_pred = model.predict(np.array(Xtest_split).reshape((Xtest_split.shape[0], Xtest_split.shape[1], n_features)),verbose=0)


    lstm_date_array.extend(test_splits[i]['date'])
    lstm_y_test_array.extend(test_splits[i]['Predict close'])
    lstm_y_test_pred_array.extend((ytest_pred[:,0].flatten()))
    lstm_batch_id_array.extend([i]*len(test_splits[i]))
    
    lstm_residuals.extend(ytest_split-ytest_pred[:,0])
    
    MAE_train = mean_absolute_error(ytrain_split,ytrain_pred[:,0])
    RMSE_train = mean_squared_error(ytrain_split,ytrain_pred[:,0],squared=False)
    MAE_test = mean_absolute_error(ytest_split,ytest_pred[:,0])
    RMSE_test = mean_squared_error(ytest_split,ytest_pred[:,0],squared=False)
    
    lstm_batch_id_array_result.append(i)
    lstm_batch_mae_train_array.append(MAE_train)
    lstm_batch_rmse_train_array.append(RMSE_train)
    lstm_batch_mae_test_array.append(MAE_test)
    lstm_batch_rmse_test_array.append(RMSE_test)


# In[44]:


lstm_result_test_df = pd.DataFrame()
lstm_result_test_df['paquet'] = lstm_batch_id_array
lstm_result_test_df['Date'] = lstm_date_array
lstm_result_test_df['y_test'] = lstm_y_test_array
lstm_result_test_df['y_test_pred'] = lstm_y_test_pred_array
lstm_y_test_array = lstm_result_test_df['y_test']
lstm_y_test_pred_array = lstm_result_test_df['y_test_pred']

lstm_result_metrics_df = pd.DataFrame()
lstm_result_metrics_df['paquet'] = lstm_batch_id_array_result
lstm_result_metrics_df['MAE_train'] = lstm_batch_mae_train_array
lstm_result_metrics_df['RMSE_train'] = lstm_batch_rmse_train_array
lstm_result_metrics_df['MAE_test'] = lstm_batch_mae_test_array
lstm_result_metrics_df['RMSE_test'] = lstm_batch_mae_test_array

pd.DataFrame(data = {"résultats LSTM model":lstm_result_metrics_df.mean()}).drop(['paquet'],axis=0)


# In[47]:


plt.figure(figsize=(30,10))
plt.plot(lstm_date_array,lstm_result_test_df["y_test"],label='actual')
plt.plot(lstm_date_array,lstm_result_test_df["y_test_pred"],label='predicted')
plt.legend()
plt.grid()
plt.show()


# In[51]:


# Le score R² 
sumofsquares = 0
sumofresiduals = 0
for i in range(len(lstm_result_test_df["y_test"])) :
    sumofsquares += (lstm_result_test_df["y_test"][i] - np.mean(lstm_result_test_df["y_test_pred"])) ** 2
    sumofresiduals += (lstm_result_test_df["y_test"][i] - lstm_result_test_df["y_test_pred"][i]) **2
    
score  = 1 - (sumofresiduals/sumofsquares)
print(score)


# In[52]:


pyplot.figure(figsize=(15,5))
_residuals = pd.DataFrame(lstm_residuals)
_residuals.hist(figsize=(10,5),bins = 40)
plt.title("Histogramme des résidus")
pyplot.show()


# In[53]:


measurements = lstm_residuals
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[54]:


plt.figure(figsize=(10,7))
plt.scatter(lstm_result_test_df["y_test"].values, lstm_residuals,facecolors='none', edgecolors='b')
plt.axhline(y=0)
plt.xlabel("Valeurs prévues")
plt.ylabel("Résidus")
plt.title("Valeurs prévues vs Résidus")
plt.show()


# In[55]:


plt.figure(figsize=(30,7))
plt.plot(lstm_date_array,lstm_residuals)


# In[56]:


durbin_watson(np.array(lstm_residuals))

