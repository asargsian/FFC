import pandas as pd
import numpy as np
from sklearn import preprocessing
import sklearn.ensemble
from fancyimpute import KNN
import lime
import lime.lime_tabular
import pylab as pl
from sklearn.tree import export_graphviz as eg
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import statsmodels.api as sm
from itertools import chain


background = pd.read_csv("background.csv")

#Preprocessing
background[background < 0] = np.nan
background = background.dropna(axis=1, how='all')
background = background.replace('Other', np.nan)
values = background['hv5_ppvtae'].unique()
for i in range(1,len(values)):
    parts = values[i].split('-')
    temp = int(parts[0]) + float(parts[1])/12
    background['hv5_ppvtae'] = background['hv5_ppvtae'].replace(values[i], temp)
values = background['hv5_wj9ae'].unique()
for i in range(1, len(values)):
    if '-' in str(values[i]):
        parts = values[i].split('-')
        if '<' in str(parts[0]) or '>' in str(parts[0]):
            parts[0] = parts[0][1:]
        temp = int(parts[0]) + float(parts[1])/12
        background['hv5_wj9ae'] = background['hv5_wj9ae'].replace(values[i], temp)
values = background['hv5_wj10ae'].unique()
for i in range(1, len(values)):
    if '-' in str(values[i]):
        parts = values[i].split('-')
        if '<' in str(parts[0]) or '>' in str(parts[0]):
            parts[0] = parts[0][1:]
        temp = int(parts[0]) + float(parts[1])/12
        background['hv5_wj10ae'] = background['hv5_wj10ae'].replace(values[i], temp)
values = background['hv5_dsae'].unique()
for i in range(1, len(values)):
    if ':' in str(values[i]):
        parts = values[i].split(':')
        if '<' in str(parts[0]) or '>' in str(parts[0]):
            parts[0] = parts[0][1:]
        temp = int(parts[0]) + float(parts[1])/12
        background['hv5_dsae'] = background['hv5_dsae'].replace(values[i], temp)
values = background['hv5_wj9pr'].unique()
for i in range(1, len(values)):
    if '<' in str(values[i]) or '>' in str(values[i]):
        temp = float(str(values[i])[1:])
        background['hv5_wj9pr'] = background['hv5_wj9pr'].replace(values[i], temp)
values = background['hv5_ppvtpr'].unique()
for i in range(1, len(values)):
    if '<' in str(values[i]) or '>' in str(values[i]):
        temp = float(str(values[i])[1:])
        background['hv5_ppvtpr'] = background['hv5_ppvtpr'].replace(values[i], temp)
values = background['hv5_wj10pr'].unique()
for i in range(1, len(values)):
    if '<' in str(values[i]) or '>' in str(values[i]):
        temp = float(str(values[i])[1:])
        background['hv5_wj10pr'] = background['hv5_wj10pr'].replace(values[i], temp)

values = background['hv4j5a_ot'].unique()
for i in range(1, len(values)):
    if '.' in str(values[i]) or 'M' in str(values[i]):
        temp = np.nan
        background['hv4j5a_ot'] = background['hv4j5a_ot'].replace(values[i], temp)

nunique = background.apply(pd.Series.nunique);
cols_to_drop = nunique[nunique == 1].index;
background = background.drop(cols_to_drop, axis=1);
nunique = background.apply(pd.Series.nunique);
background['cf4fint'] = background['cf4fint'].astype('category');
myrange = len(background['cf4fint'].value_counts())
background['cf4fint'] = background['cf4fint'].cat.rename_categories(list(range(1,myrange+1)))

background['hv5_wj9ae'] = background['hv5_wj9ae'].astype('float')

#at least 400 non-NAN values for each column
background = background.dropna(axis=1, how='all')
background = background.dropna(axis=1, thresh = 400)

#Missing value imputation with KNN
background_filled_knn = KNN(k=100).complete(background)
background_filled_knn = pd.DataFrame(background_filled_knn, columns=[list(background)])
background_filled_knn.to_csv('background_filled_knn.csv')

train = pd.read_csv('train.csv');
train = train[['challengeID', 'gpa']]
train = train.dropna()
training = pd.concat([background_filled_knn, train],axis=1, join='inner');

#training.to_csv('concated_train_background.csv')




#getting features from rlasso with tuning
feature_set = []
from sklearn.linear_model import RandomizedLasso
params = [0.004, 0.0000004]
predictors = training.iloc[:, 1:len(list(training))-1]
targets = training['gpa']
pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.3)

for i in params:
    rlasso = RandomizedLasso(alpha=i)
    rlasso.fit(pred_train, tar_train)
    indices = np.argsort(np.abs(rlasso.scores_))[::-1]
    smbo_rlasso = pred_train.iloc[:, indices[:500]]
    feature_set.append(list(smbo_rlasso)[1:])

#getting features from Extra Trees Regressor
from sklearn.ensemble import ExtraTreesRegressor
forest = ExtraTreesRegressor(n_estimators=500,
                              random_state=0)

forest.fit(pred_train, tar_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
result = pred_train.iloc[:, indices[:500]]

feature_set.append(list(result))

#intersect three sets of features obtained above
features = list(set(feature_set[0]).intersection(*feature_set[1:]))


if len(features) <=30:
    print "WARNING: the selected set of features most probably is not a proper one, and may result in low prediction accuracy. \n Please stop the experiment and rerun. If this message persists, we recommend omitting the feature selection step and instead using the features in line 138."

recommended = ['m1e3b4', 'k5a2f', 'p5i23', 'hv4l18', 'hv4l17', 'm4i0l', 'f4h2a_2',
               'p5q3u', 'm2g6d1', 'm2f2b3', 'o5c6', 'm5e1n', 'm1i1', 'm1i3',
               'm5f24', 'p5q3g', 'm5f23c', 'o5c9', 'f5g35', 'm1e3b5', 'k5g2m',
               'k5g2h', 'hv3m2c', 'f2g1a', 't5c13b', 'p5i35', 't5b3e', 'p5m1',
               't5c8b', 'm5b30', 'm4a14', 'm4a15', 'hv5_ppvtpr', 'p5l15a_5',
               'hv4p7c', 'hv3m23', 'f3h6a', 'm1d2c', 'm5e8_5', 'hv3c5', 't5e19e',
               'f5e1h', 'p5l18', 'cf1edu', 'hv3r9', 'm2h14a4', 'm1e1c1', 't5b1w',
               'm3k14b4', 't5b1u', 'f5k17g', 'hv5_wj10pr', 'f4i5a', 'm3i18',
               'm3h6', 'f1b8', 'k5g2d', 'f2d7g', 'hv5_ppvtss', 'm1b4g', 'k5a1a',
               'hv3m21', 'p5l5', 'f3b3', 'p5q3bw', 'f5k2a', 'hv3mmis_wt',
               'hv5_wj10ae', 'f1b20']

#using Random Forest for prediction
from sklearn.ensemble import RandomForestRegressor

max_score = 0

scores = []
for i in range(200,400,5):
    predictors = training[features]
    targets = training['gpa']
    pred_train, pred_test, tar_train, tar_test  = train_test_split(predictors, targets, test_size=.2)
    reg=RandomForestRegressor(n_estimators=i, max_depth=None, min_samples_split=3)
    reg=reg.fit(pred_train,tar_train)
    predictions = reg.predict(pred_test)
    score = reg.score(pred_test, tar_test)
    scores.append((i, score))
    print (i, score)
    if score >= max_score:
        all_pred = reg.predict(predictors[features])
        pd.DataFrame(all_pred).to_csv('final_check_intersection_pred_abv_score'+str(score)[:10]+'.csv')



#reading names of features
import re
featdict=dict(re.compile('---\r\n(\S+).+?(\S.+?)\r\n').findall(open("codebook_FFChallenge.txt").read()))
names=[featdict[tmp] for tmp in features]

#Decision tree

gpas = np.arange(1, 4.25, 0.25)
for i in gpas:
    if i < 2.75:
        training.loc[training['gpa']==i, 'gpa'] = 'Low'
    elif i < 3.25 and i >= 2.75:
        training.loc[training['gpa']==i, 'gpa'] = 'Mid'
    elif i <= 4 and i >= 3.15:
        training.loc[training['gpa']==i, 'gpa'] = 'Top'
training.gpa = training.gpa.astype('category')

test = training.values
pred = test[:, :-2]
target = test[:, -1]


train,test,labels_train,labels_test=pred,pred,target,target
train_hl=train[(labels_train=='Top') | (labels_train=='Low')]
labels_train_hl=labels_train[(labels_train=='Top') | (labels_train=='Low')]




dt=sklearn.tree.DecisionTreeClassifier(max_depth=5)
dt.fit(train_hl,labels_train_hl)

len(pl.find(training.gpa=='Low'))

pd.value_counts(labels_train_hl)


# Shortened names of important features


names_mod=deepcopy(names)
names_mod[7]="Disobedient at home"
names_mod[53]='Eligible for welfare in past year'
names_mod[47]='Child attends to instructions'
names_mod[24]="Child's science and social studies"
[tmp for tmp in range(len(names)) if 'C13B' in names[tmp]]

eg(dt,out_file='l1_tree.dot',class_names=('Low','Top'),rounded=True,feature_names=names_mod,max_depth=2,filled=True,impurity=False)
eg(dt,out_file='l2_tree.dot',class_names=('Low','Top'),rounded=True,feature_names=names_mod,max_depth=3,filled=True,impurity=False)


#LIME

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=550)
rf.fit(train, labels_train)

explainer = lime.lime_tabular.LimeTabularExplainer(train_hl, feature_names=names, class_names=['Low','Top'], discretize_continuous=False)

n=train_hl.shape[0]
def verboserange(num):
    for count in range(num):
        print("Count="+str(count))
        yield count
exps = [explainer.explain_instance(train_hl[tmp], rf.predict_proba, labels=(0,), num_features=20) for tmp in verboserange(n)]

# class one:
exps_1 = [explainer.explain_instance(train_hl[tmp], rf.predict_proba, labels=(2,), num_features=20) for tmp in verboserange(n)]

import pickle
pickle.dump((train_hl,labels_train_hl,names),open("training_featurematrix_andfeaturenames.pkl",'w+'))
pickle.dump(exps,open("explanations.pkl",'w+'))
pickle.dump(exps_1,open("explanations_label2.pkl",'w+'))



#K-Means clustering

# Converts each list into a fixed feature vector for clustering

# Use convention from original (label=0) file

exps0 = pickle.load(open("./explanations.pkl"))
featlist = list(set(chain(*[[tmp[0] for tmp in e.as_list(label=0)] for e in exps0])))
featinds = {featlist[ind]: ind for ind in range(len(featlist))}


def CreateVec(inlist):
    res = [0] * len(featlist)

    for tmp in inlist:
        # print tmp
        res[featinds[tmp[0]]] = tmp[1]
    return res

exps=pickle.load(open("explanations_label2.pkl"))
inlists=[e.as_list(label=2) for e in exps]
expmat=np.array([CreateVec(tmp) for tmp in inlists])

km=KMeans(n_clusters=5,random_state=100)
km.fit(expmat)
np.savetxt('km_labels.csv',km.labels_, delimiter=',', header='0')

# Presenting KMeans results

inds = pd.value_counts(km.labels_)
print(inds)

cents = km.cluster_centers_
pl.figure(figsize=(6, 4), tight_layout=True)
for count in range(len(inds)):
    pl.subplot(km.n_clusters, 1, count + 1)
    pl.plot(cents[inds.index[count], :])
    pl.grid(True)

print(pd.value_counts(km.predict(expmat)))

#logistic regression coefficients
lr=LogisticRegression(penalty='l1',C=.01).fit(train_hl,labels_train_hl)


pl.close()
pl.bar(range(69),lr.coef_[0])
pl.plot([-5,75],[0,0])
pl.axis('tight')
pl.xlabel('Feature')
pl.ylabel('Logistic Regression Coefficient')
pl.show()
pl.savefig('LR_c.01.png')



#logistic regression
# utility functions

def PlotCoefs(coefs,xl='Feature',yl='Logistic Regression Coefficient'):
    pl.bar(range(len(coefs)),coefs)
    pl.plot([-2,len(coefs)+5],[0,0])
    pl.axis('tight')
    pl.xlabel(xl)
    pl.ylabel(yl)

#getting clusters along with labels
labels = pd.read_csv('km_labels.csv')['# 0'].values.tolist()

train = pickle.load(open('training_featurematrix_andfeaturenames.pkl'))

data_all = pd.DataFrame(train[0], columns=train[2])

data_all = data_all.astype('float64')
data_all['gpa'] = train[1]
#converting to 0 if Low and 1 if Top
data_all['gpa'] = np.where(data_all['gpa'] == 'Low', 0,1)

#functions for logistic regression
def regress(cl_id):
    idx = [idx for (idx, item) in enumerate(labels) if item == cl_id]
    data = data_all.loc[idx]
    train_cols = data.columns[:-1]
    # logreg = LR(C=1e5)
    logreg = LR(C=.1, penalty='l1')
    result = logreg.fit(data[train_cols], data['gpa'])
    scores, pvalues = chi2(data[train_cols], data['gpa'])
    return [cl_id, result, pvalues]


def smregress(cl_id):
    idx = [idx for (idx, item) in enumerate(labels) if item == cl_id]
    data = data_all.loc[idx]
    train_cols = data.columns[:-1]
    mod = sm.Logit(data['gpa'], data[train_cols])
    return mod.fit(method='bfgs')


def smregress_0_2():
    idx = [idx for (idx, item) in enumerate(labels) if item in [0, 2]]
    data = data_all.loc[idx]
    train_cols = data.columns[:-1]
    mod = sm.Logit(data['gpa'], data[train_cols])
    return mod.fit(method='bfgs')


# Clusters
significants = []

for tmp in [[0, 2], [1], [3], [4]]:
    idx = [idx for (idx, item) in enumerate(labels) if item in tmp]
    data = data_all.loc[idx]
    train_cols = data.columns[:-1]
    mod0 = sm.Logit(data['gpa'], data[train_cols])
    coefs = mod0.fit_regularized(method='l1', alpha=1.5)

    # Getting significant, sorted coeffs
    betas = np.abs(coefs.params[coefs.pvalues < .05])
    inds = np.argsort(betas)[::-1]
    sigcoefs = data[train_cols].columns[coefs.pvalues < .05][inds]

    # Generating/populating output
    significants.append(sigcoefs)
    # print coefs

# Printing
print ('\n'.join(['* ' + ', '.join(tmp) for tmp in significants]))


