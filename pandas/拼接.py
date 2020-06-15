#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_dict = {'id':[1,2,3,4,5,6],
             'data':[1,1,2,3,3,3],
             'data2':[1,1,1,1,1,1]}
test_dict_df = pd.DataFrame(test_dict)


Merchant_coupon_consume = test_dict_df.groupby(by='data')
Merchant_df = Merchant_coupon_consume.size().reset_index(name='dataNumber')
temp = Merchant_coupon_consume.aggregate({'id':np.sum})
Merchant_df = pd.merge(Merchant_df, temp, how='left', on='data')
Merchant_df

# %%
test = pd.merge(test_dict_df, Merchant_df, how='left', on='data')
test


# %%
temp.columns = ['id2']
temp
# %%
