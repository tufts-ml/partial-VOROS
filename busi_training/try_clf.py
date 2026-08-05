import os
import numpy as np
import pandas as pd
import torch
import sys
import ast # to convert list to string
import copy
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from collections import OrderedDict

from warnings import filterwarnings
filterwarnings('ignore')
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) 
sys.path.append(parent_dir)
from metrics_jax import pvoros_score
from metrics import voros_score

# DATA_DIR = '/cluster/tufts/hugheslab/datasets/BUSI'

def load_dataset(df, ycol='is_malignant'):
    X = torch.vstack([torch.load(path).float() 
        for path in df['path'].to_list()]).detach().numpy()
    y = torch.tensor(df[ycol].to_list()).detach().numpy()
    return X, y

def create_xy_train_test(labels_path, random_state):
    # Load labels.csv created by our script
    df = pd.read_csv(os.path.join(labels_path, 'labels.csv'), index_col='study_id')
   
    train_df, test_df = train_test_split(
        df, train_size=0.8, random_state=random_state % 100)
        
    # Load data
    X_train, y_train = load_dataset(train_df)
    X_test, y_test = load_dataset(test_df)

    print(y_train[:5], y_train[-5:])
    print("train prev %.3f" % (np.mean(y_train)))
    print("test  prev %.3f" % (np.mean(y_test)))

    return (X_train, y_train), (X_test, y_test)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_state", type=int, default=32)
    args = parser.parse_args()
    random_state = args.random_state

    (X_train, y_train), (X_test, y_test) = create_xy_train_test(
        '/cluster/tufts/hugheslab/datasets/BUSI/kaggle_version/ViT_embeddings',
        random_state=random_state)

    clf = LogisticRegression(solver='lbfgs')
    clf_kw_grid = OrderedDict([
        ('clf__C', np.logspace(-6, 6, 25)),
        ('clf__max_iter', [4, 16, 32, 64]),
        ])
    scaler = MinMaxScaler()
    pipe = Pipeline(steps=[
        ("scaler", scaler), 
        ("clf", clf),
        ])

    # Hyperparam grid to search
    param_grid = OrderedDict()
    param_grid.update(**clf_kw_grid)

    print('Fitting classifier with C tuning')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        scoring='roc_auc',
        refit=True,
        return_train_score=True,
        verbose=3)
    search.fit(X_train, y_train)

    cv_df = pd.DataFrame(search.cv_results_)
    search_param_list = ['param_' + k for k in param_grid.keys()]
    cv_df.sort_values(search_param_list, inplace=True)
    cv_df = cv_df.reset_index(drop=True).copy()
    cv_df['score_metric_name'] = 'roc_auc'
    result_cols = [
        'mean_test_score',
        'split0_test_score',
        'split1_test_score',
        'split2_test_score',
        'split3_test_score',
        'split4_test_score',
        'mean_train_score',
        'split0_train_score',
        'split1_train_score',
        'split2_train_score',
        'split3_train_score',
        'split4_train_score',
        ]
    print(cv_df[search_param_list + ['mean_test_score']])

    be = copy.deepcopy(search.best_estimator_)
    print(be.named_steps['clf'].__repr__())

    print('Hyperparameters of the best estimator : ')
    best_params_keys = [k for k in param_grid.keys()]
    for kk in best_params_keys:
        print('%s : %s'%(kk, search.best_estimator_.get_params()[kk]))
    # Verify these best values by double-checking CV results df
    rowid = cv_df['mean_test_score'].argmax()
    bestrowdf = cv_df.iloc[rowid:rowid+1][
        search_param_list+ ['mean_train_score', 'mean_test_score']]
    print(bestrowdf[search_param_list].T.to_string(header=False))

    print("ROC AUC during K-fold CV")
    print(bestrowdf[['mean_train_score', 'mean_test_score']].T.to_string(header=False))
    
    # Evaluate predictions of the ultimate model
    x_tr_MF = X_train
    ytrue_tr_M = y_train
    x_te_NF = X_test
    ytrue_te_N = y_test

    yproba1_tr_M = be.predict_proba(x_tr_MF)[:, 1]
    yproba1_te_N = be.predict_proba(x_te_NF)[:, 1]

    print("ROC AUC on final best estimator (after refit)")
    print('train : %.3f' % roc_auc_score(
        ytrue_tr_M, yproba1_tr_M))
    print('test : %.3f' % roc_auc_score(
        ytrue_te_N, yproba1_te_N))


    print("VOROS score on final best estimator (after refit)")
    print('train : %.3f' % pvoros_score(
            ytrue_tr_M, yproba1_tr_M, 1e-6, 1.0, 0, 1e6))
    # print(voros_score(ytrue_tr_M, yproba1_tr_M, 0, 1))
    print('test : %.3f' % pvoros_score(
            ytrue_te_N, yproba1_te_N, 1e-6, 1.0, 0, 1e6))
    # print(voros_score(ytrue_te_N, yproba1_te_N, 0, 1))


    # print('V(0, 1/3) : %.3f' % pvoros_score(
    #             ytrue_tr_M, yproba1_tr_M, 1e-6, 1.0, 0, 1/3))
    # print('V(1/3, 2/3) : %.3f' % pvoros_score(
    #             ytrue_tr_M, yproba1_tr_M, 1e-6, 1.0, 1/3, 2/3))
    # print('V(2/3, 1) : %.3f' % pvoros_score(
    #             ytrue_tr_M, yproba1_tr_M, 1e-6, 1.0, 2/3, 1))
    
    

