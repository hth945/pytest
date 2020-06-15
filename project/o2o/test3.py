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

dfoff = pd.read_csv(path+'ccf_offline_stage1_train.csv')
print(dfoff.tail(2))
print(dfoff.dtypes)

df1off = dfoff[pd.notnull(dfoff['Discount_rate'])]
#%%
df1off.groupby(by=['User_id', 'Date_received', 'Date']).size()

# %%



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
data['dateBuy'] = df1off['Date']
data['User_id'] = df1off['User_id']  # 用户id
data['Merchant_id'] = df1off['Merchant_id']  # 商家id
data['isBuy'] = pd.notnull(data['dateBuy']).astype(int)
data

#%%

user_coupon_consume = data.groupby(by='User_id')
user_df = user_coupon_consume.size().reset_index(name='RecNumber')
temp = user_coupon_consume.aggregate({'isBuy':np.sum})
temp.columns = ['recIsBuy']
user_df = pd.merge(user_df, temp, how='left', on='User_id')
user_df['recBuyProb'] = user_df.apply(lambda row:row['recIsBuy'] / row['RecNumber'], axis = 1)
user_df
#%%
user_df.to_csv(path+'user_df.csv')
# %%
Merchant_coupon_consume = data.groupby(by='Merchant_id')
Merchant_df = Merchant_coupon_consume.size().reset_index(name='sendNumber')
temp = Merchant_coupon_consume.aggregate({'isBuy':np.sum})
temp.columns = ['sendIsBuy']
Merchant_df = pd.merge(Merchant_df, temp, how='left', on='Merchant_id')
Merchant_df['sendBuyProb'] = Merchant_df.apply(lambda row: row['sendIsBuy'] / row['sendNumber'], axis = 1)
Merchant_df
#%%
Merchant_df.to_csv(path+'Merchant_df.csv')
# %%
data2 = pd.merge(data, user_df, how='left', on='User_id')
data2 = pd.merge(data2, Merchant_df, how='left', on='Merchant_id')
data2
# %%


# trainx = data2.loc[:, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type','RecNumber','recIsBuy','recBuyProb','sendNumber','sendIsBuy','sendBuyProb']].values
# trainy = data2.loc[:,  ['isBuy']].values
# testx = data2.loc[:, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type','RecNumber','recIsBuy','recBuyProb','sendNumber','sendIsBuy','sendBuyProb']].values
# testy = data2.loc[:,  ['isBuy']].values

trainx = data2.loc[:, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type','recBuyProb','sendBuyProb']].values
trainy = data2.loc[:,  ['isBuy']].values
testx = data2.loc[:, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type','recBuyProb','sendBuyProb']].values
testy = data2.loc[:,  ['isBuy']].values

# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

trainDataset = tf.data.Dataset.from_tensor_slices((trainx,trainy)).shuffle(1024).repeat().batch(256)
testDataset = tf.data.Dataset.from_tensor_slices((testx,testy)).repeat().batch(256)


#%%
model = tf.keras.Sequential([tf.keras.layers.Dense(30, input_shape=(8,), activation='relu'),
                            tf.keras.layers.Dense(30, activation='relu'),
                             tf.keras.layers.Dense(2,activation='softmax')]
)

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

#%%

model.fit(trainDataset, epochs=10,steps_per_epoch=1000,
            validation_data=testDataset, validation_steps=200)

# %%
print(model.predict(trainx[0:20]))
print(trainy[0:20])

# %%
testy[0:3]

# %%

for x,y in trainDataset.take(1):
    print(x)
    print(y)


#%%
# %%
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve

pred_ = model.predict(testx)

# %%
fpr, tpr, thresholds = roc_curve(testy, pred_[:,1], pos_label=1)
auc(fpr, tpr)

# %%

data.to_csv(path+'trainDataset.csv')

# %%
data2.to_csv(path+'trainDataset2.csv')
#%%
path = '../../dataAndModel/model/o2o/'
model.save(path+"model.h5")
# %%
