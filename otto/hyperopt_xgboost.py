from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing

import numpy as np
import pandas as pd

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import sys
# The path to XGBoost wrappers goes here
sys.path.append('C:\\Users\\Amine\\Documents\\GitHub\\xgboost\\wrapper')
import xgboost as xgb


def load_train():
    train = pd.read_csv('../data/train.csv')
    labels = train.target.values
    lbl_enc = preprocessing.LabelEncoder()
    labels = lbl_enc.fit_transform(labels)
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)
    return train.values, labels.astype('int32')


def load_test():
    test = pd.read_csv('../data/test.csv')
    test = test.drop('id', axis=1)
    return test.values


def write_submission(preds, output):
    sample = pd.read_csv('../data/sampleSubmission.csv')
    preds = pd.DataFrame(
        preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv(output, index_label='id')


def score(params):
    print "Training with params : "
    print params
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    # watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid).reshape((X_test.shape[0], 9))
    score = log_loss(y_test, predictions)
    print "\tScore {0}\n\n".format(score)
    return {'loss': score, 'status': STATUS_OK}


def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : hp.quniform('max_depth', 1, 13, 1),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'num_class' : 9,
             'eval_metric': 'mlogloss',
             'objective': 'multi:softprob',
             'nthread' : 6,
             'silent' : 1
             }

    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)

    print best


X, y = load_train()
print "Splitting data into train and valid ...\n\n"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234)

#Trials object where the history of search will be stored
trials = Trials()

optimize(trials)