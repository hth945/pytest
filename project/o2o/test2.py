#%%
path = '../../dataAndModel/data/o2o/'

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from sklearn.linear_model import SGDClassifier, LogisticRegression

dfoff = pd.read_csv(path+'ccf_offline_stage1_train.csv')
print(dfoff.tail(2))
print(dfoff.dtypes)

df1off = dfoff[pd.notnull(dfoff['Discount_rate'])]

#%%
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
data['isBuy'] = pd.notnull(data['dateBuy']).astype(int)
data
#%%
data[['rate','rateMan']].plot()

# %%
trainx = data.loc[data['dateReceived'] < 20160516, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type']].values
trainy = data.loc[data['dateReceived'] < 20160516,  ['isBuy']].values
testx = data.loc[data['dateReceived'] >= 20160516, ['rate', 'rateMan', 'rateJian','Distance','weekday','weekday_type']].values
testy = data.loc[data['dateReceived'] >= 20160516,  ['isBuy']].values


# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

trainDataset = tf.data.Dataset.from_tensor_slices((trainx,trainy)).shuffle(7).repeat().batch(256)
testDataset = tf.data.Dataset.from_tensor_slices((testx,testy)).repeat().batch(256)


#%%
model = tf.keras.Sequential([tf.keras.layers.Dense(30, input_shape=(6,), activation='relu'),
                            tf.keras.layers.Dense(30, activation='relu'),
                             tf.keras.layers.Dense(2,activation='softmax')]
)

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
)

#%%

model.fit(trainDataset, epochs=10,steps_per_epoch=2000,
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


# %%
