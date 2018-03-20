
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


# In[2]:


dtype={'is_booking':bool,
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

dtype1={'srch_ci' : np.str_,
        'srch_co' : np.str_,
        'srch_adults_cnt' : np.int32,
        'srch_children_cnt' : np.int32,
        'srch_rm_cnt' : np.int32,
        'srch_destination_id':np.str_,
        'user_location_country' : np.str_,
        'user_location_region' : np.str_,
        'user_location_city' : np.str_,
        'orig_destination_distance':np.float64,
        'date_time':np.str_,
        'hotel_market':np.str_}


# In[3]:


# load datasets
# test = pd.read_csv("test.csv")
# train = pd.read_csv("train.csv")
destinations = pd.read_csv("destinations.csv")
train = pd.read_csv('train.csv',dtype=dtype, usecols=dtype, parse_dates=['date_time'] ,sep=',')
test = pd.read_csv('test.csv',dtype=dtype1,usecols=dtype1,parse_dates=['date_time'] ,sep=',')


# In[5]:


# get year and month attributes from date_time
train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month

test["year"] = test["date_time"].dt.year
test["month"] = test["date_time"].dt.month

m=train.orig_destination_distance.mean()
train.orig_destination_distance.fillna(m, inplace=True)

n=test.orig_destination_distance.mean()
test.orig_destination_distance.fillna(m, inplace=True)


# In[8]:


train = train.query('year==2013 | (year==2014 & month < 8)')


# In[9]:


train2 = train
test2 = test


# In[10]:


train2.shape


# # Baseline

# In[28]:


# BASE CASE (SIMPLE ALGO)

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)
predictions = [most_common_clusters for i in range(test2.shape[0])]


# In[28]:


write_p = [" ".join([str(l) for l in p]) for p in predictions]
write_frame = ["{0},{1}".format(test2["id"][i], write_p[i]) for i in range(len(predictions))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("base-predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))


# # Feature Engineering

# In[11]:


# Generate features from destinations
pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]


# In[12]:


# Feature Engineering
# Generate new date features based on date_time, srch_ci, and srch_co.
# Remove non-numeric columns like date_time.
# Add in features from dest_small.
# Replace any missing values with -1.
# The above will calculate features such as length of stay, check in day, and check out month.
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

df = calc_fast_features(train2)
df.fillna(-1, inplace=True)
predictors = [c for c in df.columns if c not in ["hotel_cluster", "is_booking"]]


# In[13]:


t2 = calc_fast_features(test2)
t2.fillna(-1, inplace=True)
testPredictors = [c for c in t2.columns if c not in ["hotel_cluster"]]


# # Predict based on model

# In[20]:


def predict_test_file(trained_model, test_data):
    y_predicted = []
    for i in range(6):
        print 'batch',i+1, '/7'
        y_pred = trained_model.predict_proba(test_data.iloc[i*450000:(i+1)*450000,:])
        top_5 = y_pred.argsort(axis=1)[:,-5:]
        y_predicted.append(top_5)
    dict_cluster = {}
    print 'Getting cluster names'
    for (k,v) in enumerate(trained_model.classes_):
        dict_cluster[k] = v
    print 'Translating to hotel clusters'
    b = []
    for i in np.vstack(y_predicted).flatten():
        b.append(dict_cluster.get(i))
    predict_class=np.array(b).reshape(np.vstack(y_predicted).shape)
    predict_class=map(lambda x: ' '.join(map(str,x)), predict_class)
    print 'Creating submit file'
    df_submission_sample = pd.DataFrame.from_csv('sample_submission.csv')
    df_submission_sample['hotel_cluster'] = predict_class
    df_submission_sample.to_csv('naive-bayes.csv')
    print 'Done. naive-bayes.csv ready in your folder.'


# # Random Forest

# In[15]:


## Random forest
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
clf.fit(df[predictors], df['hotel_cluster'])


# # Random Forest Classifier predict on all test data

# In[16]:


predict_test_file(clf, t2)


# In[ ]:


# write_p = [" ".join([str(l) for l in p]) for p in predictions]
# write_frame = ["{0},{1}".format(test2["id"][i], write_p[i]) for i in range(len(predictions))]
# write_frame = ["id,hotel_cluster"] + write_frame
# with open("random-forest-predictions.csv", "w+") as f:
#     f.write("\n".join(write_frame))


# # SGD Classifier

# In[17]:


from sklearn import linear_model

clf = linear_model.SGDClassifier(loss='log', n_jobs=-1, alpha=0.0000025, verbose=0)
clf.fit(df[predictors], df['hotel_cluster'])


# In[19]:


predict_test_file(clf, t2)


# # Naive Bayes

# In[21]:


from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha=1.0)
clf.partial_fit(df[predictors], df['hotel_cluster'], classes=np.arange(100))


# In[22]:


predict_test_file(clf, t2)


# # Improvements

# In[24]:


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


# In[25]:


import operator

cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top

preds = []
for index, row in test2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])


# In[26]:


# matching users
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


# In[29]:


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


# In[33]:


full_preds


# In[38]:


write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(i, write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_cluster"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))

