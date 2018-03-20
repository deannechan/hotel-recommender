
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.decomposition import PCA
import random
import ml_metrics as metrics
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


# In[2]:


dtype={'is_booking':bool,
       'user_id':np.str_,
        'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'hotel_cluster' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}


# In[3]:


# load datasets
# test = pd.read_csv("test.csv")
# train = pd.read_csv("train.csv")
destinations = pd.read_csv("destinations.csv")
train = pd.read_csv('train.csv')#,dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',')


# In[4]:


# load datasets
# destinations = pd.read_csv("destinations.csv")
# train = pd.read_csv("train.csv")


# In[5]:


# get year and month attributes from date_time
train["date_time"] = pd.to_datetime(train["date_time"])
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month


# In[6]:


train.shape


# In[7]:


# fill in missing orig_destination_distance
m=train.orig_destination_distance.mean()
train.orig_destination_distance.fillna(m, inplace=True)


# In[8]:


# random 100000 samples
unique_users = train.user_id.unique()
print len(unique_users)


# In[9]:


# pick 100,000 users
selected_user_id = unique_users[:100000]#random.sample(unique_users,10000)
selected_train = train[train.user_id.isin(selected_user_id)]


# In[10]:


# train = 2013 | until July 2014; 
# test = After July 2014
train2 = selected_train[((selected_train.year == 2013) | ((selected_train.year == 2014) & (selected_train.month < 8)))]
test2 = selected_train[((selected_train.year == 2014) & (selected_train.month >= 8))]


# In[11]:


size = len(selected_train)
print size
print len(train2)
print len(test2)


# In[12]:


# only get those whose booking is true
test2 = test2[test2.is_booking == True]


# In[13]:


print "train2 - ", len(train2)
print "test2 - ", len(test2)


# ### Generate features from destinations csv

# In[14]:


pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]


# ### Feature Engineering
# - Generate new date features based on date_time, srch_ci, and srch_co.
# - Remove non-numeric columns like date_time.
# - Add in features from dest_small.
# - Replace any missing values with -1.
# - Calculate features such as length of stay, check in day, and check out month.

# In[15]:


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
    
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret


# In[16]:


df = calc_fast_features(train2)
df.fillna(-1, inplace=True)
predictors = [c for c in df.columns if c not in ["hotel_cluster"]]


# In[17]:


t2 = calc_fast_features(test2)
t2.fillna(-1, inplace=True)
testPredictors = [c for c in t2.columns if c not in ["hotel_cluster"]]


# ### Baseline

# In[18]:


most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
predictions = [most_common_clusters for i in range(t2.shape[0])]


# In[19]:


most_common_clusters


# In[20]:


target = [[l] for l in t2["hotel_cluster"]]
print "Baseline evaluation: ", metrics.mapk(target, predictions, k=5)


# ### Making predictions

# In[21]:


target = [[l] for l in t2["hotel_cluster"]]


# In[22]:


def predict_test_file(trained_model, test_data):
    y_predicted = []
    y_pred=clf.predict_proba(test_data[testPredictors])
    
    #take largest 5 probablities' indexes
    y_predicted=y_pred.argsort(axis=1)[:,-5:]
    
    print 'Getting cluster names'
    dict_cluster = {}
    for (k,v) in enumerate(trained_model.classes_):
        dict_cluster[k] = v
        
    print 'Translating to hotel clusters'
    b = []
    for i in np.vstack(y_predicted).flatten():
        b.append(dict_cluster.get(i))
    predict_class=np.array(b).reshape(np.vstack(y_predicted).shape)
#     predict_class=map(lambda x: ' '.join(map(str,x)), predict_class)
    print predict_class
    
#     target = [[l] for l in t2["hotel_cluster"]]
    print "Evaluation: ", metrics.mapk(target,predict_class,k=5)


# ### Random Forest

# In[23]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
clf = RandomForestClassifier(n_estimators=31,max_depth=10,random_state=123)
clf.fit(df[predictors], df['hotel_cluster'])


# In[24]:


importance = clf.feature_importances_
indices=np.argsort(importance)[::-1][:10]
plt.barh(range(10), importance[indices],color='r')
plt.yticks(range(10),df[predictors].columns[indices])
plt.xlabel('Feature Importance')
plt.show()


# In[25]:


# dict_cluster = {}
# for (k,v) in enumerate(clf.classes_):
#     dict_cluster[k] = v

# y_pred=clf.predict_proba(t2[testPredictors])
# #take largest 5 probablities' indexes
# a=y_pred.argsort(axis=1)[:,-5:]

# #take the corresonding cluster of the 5 top indices
# b = []
# for i in a.flatten():
#     b.append(dict_cluster.get(i))
    
# cluster_pred = np.array(b).reshape(a.shape)
# target = [[l] for l in t2["hotel_cluster"]]
# print "Random Forest score: ",metrics.mapk(target,cluster_pred,k=5)

predict_test_file(clf, t2)


# ### SGD Classifier

# In[39]:


from sklearn import linear_model

clf = linear_model.SGDClassifier(loss='log', n_jobs=-1, alpha=0.0000025, verbose=0)
clf.partial_fit(df[predictors], df['hotel_cluster'], classes=np.arange(100))


# In[40]:


# dict_cluster = {}
# for (k,v) in enumerate(clf.classes_):
#     dict_cluster[k] = v
    
# y_pred=clf.predict_proba(t2[testPredictors])
# #take largest 5 probablities' indexes
# a=y_pred.argsort(axis=1)[:,-5:]

# #take the corresonding cluster of the 5 top indices
# b = []
# for i in a.flatten():
#     b.append(dict_cluster.get(i))
    
# cluster_pred = np.array(b).reshape(a.shape)
# target = [[l] for l in t2["hotel_cluster"]]
# print "Stochastic Gradient Descent (SGD) score: ",metrics.mapk(target,cluster_pred,k=5)

predict_test_file(clf, t2)


# ### Naive Bayes

# In[28]:


from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=1.0)
clf.partial_fit(df[predictors], df['hotel_cluster'], classes=np.arange(100))


# In[29]:


# dict_cluster = {}
# for (k,v) in enumerate(clf.classes_):
#     dict_cluster[k] = v

# y_pred=clf.predict_proba(t2[testPredictors])
# #take largest 5 probablities' indexes
# a=y_pred.argsort(axis=1)[:,-5:]

# print "nb_a:", a 

# #take the corresonding cluster of the 5 top indices
# b = []
# for i in a.flatten():
#     b.append(dict_cluster.get(i))

# cluster_pred = np.array(b).reshape(a.shape)
# target = [[l] for l in t2["hotel_cluster"]]
# print "Naive Bayes score: ",metrics.mapk(target,cluster_pred,k=5)
predict_test_file(clf, t2)


# ### Improvements
# - Aggregate `hotel_cluster` based on `srch_destination_id`
# > find the most popular hotel clusters for each destination to predict that a user who searches for a destination is going to one of the most popular hotel clusters for that destination
# - Group training data by `search_destination_id` and `hotel_cluster`
# - Iterate each group
# > - Assign 1 point to each hotel cluster where `is_booking` is True.
# > - Assign .15 points to each hotel cluster where `is_booking` is False.
# > - Assign the score to the `srch_destination_id` / `hotel_cluster` combination in a dictionary.
# 
# #### Output:
# We'll have a dictionary where each key is a `srch_destination_id`. Each value in the dictionary will be another dictionary, containing `hotel_clusters` as keys with scores as values
# 

# In[30]:


def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = train2.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])
    
    score = bookings + .15 * clicks
    
    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score


# In[31]:


top_clusters


# ### Find top 5 clusters for each `srch_destination_id`
# - Loop through each key in top_clusters.
# - Find the top 5 clusters for that key.
# - Assign the top 5 clusters to a new dictionary, cluster_dict

# In[32]:


import operator

cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top
cluster_dict


# ### Make predictions based on destination
# - Iterate each test data
# - Extract `srch_destination_id` for the row
# - Find top clusters for that `srch_destination_id`
# - Append top clusters to preds
# 
# #### Output
# `preds` - a list of lists containing predictions

# In[33]:


preds = []
for index, row in test2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])


# In[34]:


metrics.mapk([[l] for l in test2["hotel_cluster"]], preds, k=5)


# ### Finding matching users
# Finding users in training set that matches in testing set
# - Split the training data into groups based on the match columns.
# - Loop through the testing data.
# - Create an index based on the match columns.
# - Get any matches between the testing data and the training data using the groups.

# In[35]:


match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']

groups = train2.groupby(match_cols)
    
def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(test2.shape[0]):
    exact_matches.append(generate_exact_matches(test2.iloc[i], match_cols))


# ### Combining predictions

# In[36]:


def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result
    
full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
metrics.mapk([[l] for l in test2["hotel_cluster"]], full_preds, k=5)

