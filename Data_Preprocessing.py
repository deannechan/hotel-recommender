
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


destinations = pd.read_csv("destinations.csv")
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")


# In[5]:


def calc_fast_features(df):
    df.loc[:,'date_time'] = pd.to_datetime(df["date_time"])
    df.loc[:,'srch_ci'] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
    df.loc[:,'srch_co'] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
    
    ret = pd.DataFrame(props)
    
#     ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_id", axis=1)
    return ret

df = calc_fast_features(train)
df.fillna(-1, inplace=True)


# In[26]:


train['date_time']=pd.to_datetime(train['date_time'],infer_datetime_format = True,errors='coerce')
train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')

train['month']= train['date_time'].dt.month
train['duration']=((train['srch_co']-train['srch_ci'])/np.timedelta64(1,'D')).astype(float)


# In[27]:


train.columns


# In[28]:


test_corr = test.corr()


# In[32]:


train_corr = train.corr()


# In[18]:


train.isnull().sum(axis=0)


# # __FEATURE MAP CORRELATION__

# In[34]:


seaborn.heatmap(train_corr, 
                xticklabels=train_corr.columns.values, 
                yticklabels=train_corr.columns.values, 
                cmap="RdBu", 
                vmin=-1, 
                vmax=1)


# In[35]:


train_corr


# # __HOTEL CLUSTER FREQUENCY__

# In[9]:


# train.head(5)


# In[37]:


train['year']= train['date_time'].dt.year
train2 = train.query('year==2013 | (year==2014 & month < 8)')


# In[38]:


hotel_clusters = train2['hotel_cluster']
# hotel_clusters


# In[11]:


train2['hotel_cluster'].value_counts()


# In[39]:


x_pos = np.arange(100)
performance = train2['hotel_cluster'].value_counts().sort_index()


# In[40]:


plt.bar(x_pos, performance)
plt.xlabel('Hotel Cluster')
plt.ylabel('Frequency')
 
plt.show()


# # __RELATIVE FEATURE IMPORTANCE__

# In[21]:


train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

train['srch_ci']=pd.to_datetime(train['srch_ci'],infer_datetime_format = True,errors='coerce')
train['srch_co']=pd.to_datetime(train['srch_co'],infer_datetime_format = True,errors='coerce')
train['duration']=(train['srch_co']-train['srch_ci']).astype('timedelta64[h]')
train = train.query('is_booking==True & year==2014')


# In[22]:


# Feature Importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

y=train['hotel_cluster']
X=train.drop(['hotel_cluster','is_booking', 'date_time', 'srch_co', 'srch_ci'],axis=1) # in training dataset, have clicking and booking event


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)
X_train.columns


# In[27]:


m=X_train.orig_destination_distance.mean()
X_train.orig_destination_distance.fillna(m, inplace=True)


# In[28]:


rf_tree = RandomForestClassifier(n_estimators=31,max_depth=10,random_state=123)
rf_tree.fit(X_train,y_train)


# In[29]:


importance = rf_tree.feature_importances_
indices=np.argsort(importance)[::-1][:10]


# In[30]:


importance[indices]


# In[31]:


plt.barh(range(10), importance[indices],color='r')
plt.yticks(range(10),X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.show()


# In[ ]:


from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
# display the relative importance of each attribute
# print(model.feature_importances_)


# In[ ]:


importance = model.feature_importances_
indices=np.argsort(importance)[::-1][:10]


# In[ ]:


plt.barh(range(10), importance[indices],color='r')
plt.yticks(range(10),X_train.columns[indices])
plt.xlabel('Feature Importance')
plt.show()

