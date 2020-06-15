#%%
path = '../../dataAndModel/data/o2o/'

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

dfoff = pd.read_csv(path+'ccf_offline_stage1_test_revised.csv')
print(dfoff.tail(2))
print(dfoff.dtypes)

df1off = dfoff
# %%
# 打折还是优惠券
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    if ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

def convertRateMan(row):
    if ':' in row:
        rows = row.split(':')
        return float(rows[0])
    else:
        return float(row)

def convertRateJian(row):
    if ':' in row:
        rows = row.split(':')
        return float(rows[1])
    else:
        return float(row)

def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

data = df1off[['Distance']].fillna(-1).astype(int)
data['rateType'] = df1off['Discount_rate'].apply(getDiscountType)
data['rate'] = df1off['Discount_rate'].apply(convertRate)
data['rateMan'] = df1off['Discount_rate'].apply(convertRateMan)
data['rateJian'] = df1off['Discount_rate'].apply(convertRateJian)
data['dateReceived'] = df1off['Date_received']
data['weekday'] = df1off['Date_received'].astype(str).apply(getWeekday)
data['weekday_type'] = data['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
data['User_id'] = df1off['User_id']  # 用户id
data['Merchant_id'] = df1off['Merchant_id']  # 商家id
data
#%%

user_df = pd.read_csv(path+'user_df.csv')
Merchant_df = pd.read_csv(path+'Merchant_df.csv')
# %%
data2 = pd.merge(data, user_df, how='left', on='User_id')
data2 = pd.merge(data2, Merchant_df, how='left', on='Merchant_id')
data2
# %%
testx = data2.loc[:, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type','recBuyProb','sendBuyProb']].values

# %%
pred_ = model.predict(testx)

# %%
print(dfoff.dtypes)

# %%
p = pred_[:,1]

# %%
dfoff['Probability'] = pd.Series(p)

# %%
dfoff[['User_id', 'Coupon_id', 'Date_received', 'Probability']].to_csv(path+'out.csv',index = None)

# %%
