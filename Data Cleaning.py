# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 21:22:29 2024

@author: Lili Zheng
"""


# Import modules needed to process data

import numpy as np
import pandas as pd

# Import data
dataset = pd.read_csv(FileAddress)

# Handling the text attributes
dataset['Text Column Name'].fillna('', inplace = True)

# Handling the numeric attributes
dateset['Numeric Column Name'].fillna(0, inplace = True)
dateset['Numeric Column Name'] = pd.to_numeric(dataset['Numeric Column Name']).astype('float64')

# Unify the outliers in the text attributes
dataset['Wrong Case Letter Column Name'] = dataset['Wrong Case Letter Column Name'].str.upper()
dataset['Wrong Case Letter Column Name'] = np.where(data['Wrong Case Letter Column Name'] == 'UNITED STATES',
                                                    'USA',
                                                    data['Wrong Case Letter Column Name'])

# Bad data entry
dataset['N/A Column Name'] = np.where(dataset['N/A Column Name']=='N/A','', dataset['N/A Column Name'])
dataset['NAN Column Name'] = np.where(dataset['NAN Column Name']=='Nan','', dataset['NAN Column Name'])
dataset['Null Column Name'] = np.where(dataset['Null Column Name']=='Null','', dataset['Null Column Name'])
dataset['Wrong Letter COlumn Name'] = dataset['Wrong Letter COlumn Name'].str.replace('Â', '')

# Handling outliers
dataset['Outlier Column Name'] = dataset['Outlier Column Name'].astype(float)
dataset['Outlier Column Name'] = np.where(dataset['Outlier Column Name']<=100, 0, dataset['Outlier Column Name'])

# Adding new attribute
dataset['GOB'] = dataset.apply(lambda row: row['gross']/row['budget'] if row['budget']!=0 else 0, axis=1)
