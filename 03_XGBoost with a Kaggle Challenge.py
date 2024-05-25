# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:15:43 2024

@author: Lili Zheng

Target to design a prediction model using XGBoost

Summary

I. Demand Planning Optimization Problem Statement
Forecast the demand of 50 retail stores in US

II. XGBoost for Sales Forecasting
Build a forecating model using Machine Learning

III. Demand Planning: XGBoost vs. Rolling Mean
1. Demand Planning using Rolling Mean
An initial approach using a simple formular to set the baseline
2. XGBoost vs. Rolling Mean
What is the impact of Machine Learning on Accuracy?

IV. Next Steps
1. Simulation Model with ChatGPT - "The Supply Chain Analyst"
   Implement analytics products with UI powered by GPT on ChatGPT
2. Implement Inventory Management Rules
   Combine your forecasting model with Inventory Rules to reduce stockouts
3. Sustainable Approach: Green Inventory Management
   Reduce the carbon footprint of your supply chain with smart inventory rules
4. Optimize Procurement Management with Python

Parameter introduction: https://www.cnblogs.com/TimVerion/p/11436001.html
Evaluation introduction: https://sklearn.apachecn.org/master/32/#google_vignette

"""

# =============================================================================
# Add data features
# =============================================================================

import numpy as np
import pandas as pd

# Import training and test data
train = pd.read_csv(r"D:\Python\Document From Samir Saci\Machine Learning for Retail Demand Forecasting\train.csv")
test = pd.read_csv(r"D:\Python\Document From Samir Saci\Machine Learning for Retail Demand Forecasting\test.csv")

# Dates Features
def date_features(df):
        
    # Date Features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.day
    df['dayofyear'] = df.date.dt.dayofyear
    df['dayofweek'] = df.date.dt.dayofweek
    df['weekofyear'] = df.date.dt.isocalendar().week
    
    # Additional Data Freatures
    df['day^year'] = np.log((np.log(df['dayofyear'] + 1)) ** (df['year'] - 2000))
    
    # Drop date
    # df.drop('date', axis = 1, inplace = True)
    
    return df

# Dates Ferstures for Train, Test
train, test = date_features(train), date_features(test)

# =============================================================================
# Daily, Monthly Average for Train
# =============================================================================

train['daily_avg'] = train.groupby(['item', 'store', 'dayofweek'])['sales'].transform('mean')
train['monthly_avg'] = train.groupby(['item', 'store', 'month'])['sales'].transform('mean')
train = train.dropna()

# Average sales for Day_of_week = d per Item, Store
daily_avg = train.groupby(['item', 'store', 'dayofweek'])['sales'].mean().reset_index()
# Average sales for Month = m per Item, Store
monthly_avg = train.groupby(['item', 'store', 'month'])['sales'].mean().reset_index()

# =============================================================================
# Add Daily and Monthly Averages to Test and Rolling Averages
# =============================================================================

# Merge Test with Daily Avg, Monthly Avg
def merge(df1, df2, col, col_name):
    
    df1 = pd.merge(df1, df2, how = 'left', on = None, left_on = col, right_on = col,
                   left_index = False, right_index = False, sort = False,
                   copy = True, indicator = False)
    
    df1 = df1.rename(columns = {'sales': col_name})
    
    return df1

# Add Daily_avg and Monthly_avg features to test
test = merge(test, daily_avg, ['item', 'store', 'dayofweek'], 'daily_avg')
test = merge(test, monthly_avg, ['item', 'store', 'month'], 'monthly_avg')

# Sales rolling mean sequence per item
rolling_10 = train.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis = 1)
# Update the training dataframe with new column
train['rolling_mean'] = rolling_10['sales']

# 90 last days of training rolling mean sequence added to test data
# Basically this is last 3 months of the year
rolling_last90 = train.groupby(['item', 'store'])['rolling_mean'].tail(90).copy()
test['rolling_mean'] = rolling_last90.reset_index().drop('index', axis = 1)

# Shifting rolling mean 3 months
train['rolling_mean'] = train.groupby(['item'])['rolling_mean'].shift(90) # Create a feature with rolling mean of day
train.head()

# =============================================================================
# Clean features, Training / Test Split and Run model
# =============================================================================

import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# Features Scaling (except sales)
sales_series, date_series, id_series = train['sales'], train['date'], test['id']

# Clean features highly correlated to each others
for df in [train, test]:
    df.drop(['date',
             'dayofyear',
             'weekofyear',
             'daily_avg',
             'day',
             'month',
             'item',
             'store',],
            axis = 1,
            inplace = True)

# Features Scaling
train = (train - train.mean()) / train.std()
test = (test - test.mean()) / test.std()

# Retrieve actual Sales values and ID
train['sales'] = sales_series
train['date'] = date_series
test['id'] = id_series

# df = train
df_train = train.copy()

# Train Test Split with module from sklearn.model_selection
x_train, x_test, y_train, y_test = train_test_split(df_train.drop('sales', axis = 1), df_train.pop('sales'),
                                                    random_state = 123, test_size = 0.2)

# Set up the parameters
params_baseline = {
    
    ### General parameters ###

    'booster': 'gbtree',         
    'nthread': 3,                
    'silent': False,             
    
    ### Booster parameters ###
        
    'eta': 0.1,                  
    'n_estimators': 70,
    'gamma': 2000,             
    
    'min_child_weight': 3,      
    'max_depth': 5,             
    'max_delta_step': 0,         
    'subsample': 0.7,            
    'colsample_bylevel': 1,      
    'colsample_bynode': 1,       
    'colsample_bytree': 0.7,     

    'reg_alpha': 0,             
    'reg_lambda': 100,             

    ### Learning task parameters ###

    'objective': 'reg:linear',            
    'eval_metric': 'mae',    
    'seed': 27,    
     
    }
                                               
# Train the model
regress_model = xgb.XGBRegressor(**params_baseline)
regress_model.fit(x_train.drop('date', axis = 1), y_train)

# =============================================================================
# Evaluate the output of model
# =============================================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt

y_pred = regress_model.predict(x_test.drop(['date'], axis = 1))

print("Explained Variance: ", explained_variance_score(y_test, y_pred))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))

y_test = y_test.to_frame()
y_test['predict'] = y_pred
x_test_df = pd.concat([x_test, y_test], axis = 1)
x_test_df['week'] = x_test_df['date'].dt.isocalendar().week
x_test_df['year'] = x_test_df['date'].dt.isocalendar().year
x_plot = x_test_df.groupby(['year', 'week']).agg({'sales': 'sum', 'predict':'sum', 'date': 'first'}).reset_index()

plt.plot(x_plot['date'], x_plot['predict'], color = 'r', label = 'prediction')
plt.plot(x_plot['date'], x_plot['sales'], color = 'b', label = 'actual')
plt.xlabel('day')
plt.ylabel('sales')
plt.legend(loc = 'best')
plt.show()


# =============================================================================
# Visualize the features' correlation
# =============================================================================

# Use corr function to calculate correlation matrix
train_corr = train.corr(method = 'pearson')

from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
import palettable # Palettable is a strong color brewer

# Use seaborn.heatmap to draw correlation heatmap
plt.figure(figsize = (15, 12), dpi = 100)

sns.heatmap(
            data = train_corr,
            # vmax = 0.3,
            cmap = 'Blues',
            annot = True, # show numeric text in the charts
            fmt = ".3f", # formating the numbers in the charts
            annot_kws = {'size':12, 'weight':'normal', 'color':'#253D24'}, # set up the text font and size 
            mask = np.triu(np.ones_like(train_corr, dtype = bool)), # only show the table below the diagonal
            square = True, # show the out liner
            linewidths = .5, # adjust the width of the line
            cbar_kws = {"shrink": .5}
            )

# Use seaborn.clustermap to draw clustinering
row_c = dict(zip(list(ascii_letters[:14]), plt.get_cmap('RdPu')(np.linspace(0, 1, 14))))
index_c = dict(zip(list(ascii_letters[:14]), plt.get_cmap('RdPu')(np.linspace(0, 1, 14))))
sns.set(style = 'ticks')
plt.figure(figsize = (13, 13))
sns.clustermap(
            data = train_corr,
            # vmax = 0.3,
            cmap = 'Blues',
            linewidths = 0.75,
            row_colors = pd.Series(train_corr.columns.get_level_values(None), index = train_corr.columns).map(row_c),
            col_colors = pd.Series(train_corr.columns.get_level_values(None), index = train_corr.columns).map(index_c),
            dendrogram_ratio = 0.15, 
            )

# =============================================================================
# Optimize the parameters
# =============================================================================

from sklearn.model_selection import GridSearchCV

params_adjust = {
    
    # 'eta': [0.01, 0.05, 0.1, 0.3],
    'n_estimator': [100, 500, 1000, 1500, 2000],
    # 'gamma': np.linspace(0, 1, 10),
    # 'subsample': np.linspace(0.3, 1, 10),
    # 'colsample_bytree': np.linspace(0, 1, 11),
    # 'reg_lambda': np.linspace(0, 100, 11),
    
    }

model_gs = GridSearchCV(estimator = regress_model, 
                        param_grid = params_adjust, 
                        scoring = 'r2', 
                        cv = 3)

model_gs.fit(x_train.drop(['date'], axis = 1), y_train)

# check the best result

print('Best Parameters: ', model_gs.best_params_)
print('Best model score: ', model_gs.best_score_)

params_best = {
    
    ### General parameters ###

    'booster': 'gbtree',         
    'nthread': 3,                
    'silent': False,             
    
    ### Booster parameters ###
        
    'eta': 1,                  
    'n_estimator': 500,
    'gamma': 0.1,             
    
    'min_child_weight': 3,      
    'max_depth': 10,             
    'max_delta_step': 0,         
    'subsample': 0.7,            
    'colsample_bylevel': 1,      
    'colsample_bynode': 1,       
    'colsample_bytree': 0.7,     

    'reg_alpha': 0,             
    'reg_lambda': 1,             

    ### Learning task parameters ###

    'objective': 'reg:linear',   
    'base_score': 0.5,           
    'eval_metric': 'mae',        
    'seed': 0,                   
    'scale_pos_weight': 1        
    }

regress_model_best = xgb.XGBRegressor(**params_best)
regress_model_best.fit(x_train.drop('date', axis = 1), y_train)
y_pred_best = regress_model_best.predict(x_test.drop(['date', 'actual', 'predict', 'week'], axis = 1))

print("Explained Variance: ", explained_variance_score(y_test, y_pred_best))
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred_best))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred_best))
print("R2: ", r2_score(y_test, y_pred_best))

x_test['predict'] = y_pred_best
x_plot = x_test.groupby(['year', 'week']).agg({'actual': 'sum', 'predict':'sum', 'date': 'first'}).reset_index()

plt.plot(x_plot['date'], x_plot['predict'], color = 'r', label = 'prediction')
plt.plot(x_plot['date'], x_plot['actual'], color = 'b', label = 'actual')
plt.xlabel('day')
plt.ylabel('sales')
plt.legend(loc = 'best')
plt.show()
plt.cla()
