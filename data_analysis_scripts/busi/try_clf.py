import os
import numpy as np
import pandas as pd
import torch
import sys
import ast # to convert list to string
import copy
import argparse

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from collections import OrderedDict

from warnings import filterwarnings
filterwarnings('ignore')

def load_dataset(df, key_cols=[], ycol='is_malignant'):
    X = torch.vstack([torch.load(path).float() 
        for path in df['path'].to_list()]).detach().numpy()
    y = torch.tensor(df[ycol].to_list()).detach().numpy()
    k = df[key_cols]
    return X, y, k

def create_xy_train_test(labels_path, random_state, frac_train=0.8,
        csv_basename='clean_labels_with_patient_id.csv',
        y_col='is_malignant',
        group_col='patient_id',
        key_cols=['study_id', 'basename']):

    key_cols = [group_col] + key_cols

    # Load labels.csv created by our script
    df = pd.read_csv(os.path.join(labels_path, csv_basename))

    n_splits = int(np.ceil(1 / (1 - frac_train)))
    splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=random_state % 100)
    split = splitter.split(df.index, df[y_col], groups=df[group_col])
    train_ids, test_ids = split.__next__()

    train_df = df.loc[train_ids]
    test_df = df.loc[test_ids]
    #train_df, test_df = train_test_split(
    #    df, train_size=0.8, random_state=random_state % 100)
        
    # Load data
    X_train, y_train, key_train = load_dataset(train_df, key_cols, y_col)
    X_test, y_test, key_test = load_dataset(test_df, key_cols, y_col)
    print("Train/Test split summary")
    print("----------")
    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)
    print("First 10 rows of train")
    print(key_train.head(10))
    print("First 10 rows of test")
    print(key_test.head(10))
    print("train prev %.3f" % (np.mean(y_train)))
    print("test  prev %.3f" % (np.mean(y_test)))

    return (X_train, y_train, key_train), (X_test, y_test, key_test)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str,
        default='/cluster/tufts/hugheslab/datasets/BUSI/kaggle_version/')
    parser.add_argument("--random_state", type=int, default=32)
    parser.add_argument("--group_col", type=str, default='patient_id')
    parser.add_argument('--reduced_dim', type=int, default=None)
    parser.add_argument('--n_folds_cv', type=int, default=5)
    args = parser.parse_args()
    random_state = args.random_state
    group_col = args.group_col

    (X_train, y_train, k_train), (X_test, y_test, k_test) = create_xy_train_test(
        args.dataset_path,
        group_col=group_col,
        random_state=random_state)

    clf = LogisticRegression(solver='lbfgs', random_state=random_state)
    clf_kw_grid = OrderedDict([
        ('clf__C', np.logspace(-6, 6, 25)),
        ('clf__max_iter', [4, 16, 32, 64, 1024]),
        ])
    scaler = StandardScaler()
    
    if args.reduced_dim is None:
        steps = [
            ("scaler", scaler), 
            ("clf", clf),
            ]
    else:
        steps = [
            ("scaler", scaler), 
            ("reducer", PCA(
                n_components=args.reduced_dim,
                random_state=random_state)), 
            ("clf", clf),
            ]   
    pipe = Pipeline(steps=steps)

    # Hyperparam grid to search
    param_grid = OrderedDict()
    param_grid.update(**clf_kw_grid)

    print('\nFitting classifier with tuning of C and max_iter ...')
    cv = StratifiedGroupKFold(n_splits=args.n_folds_cv, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        pipe, param_grid,
        cv=cv,
        scoring='roc_auc',
        refit=True,
        return_train_score=True,
        verbose=0)
    search.fit(X_train, y_train, groups=k_train[group_col])
    print("... complete.'")

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
    print("\nFirst 10 rows of cv results data frame")
    print("-------------")
    print(cv_df[search_param_list + ['mean_test_score']].head(10))

    be = copy.deepcopy(search.best_estimator_)
    print(be.named_steps['clf'].__repr__())
    if args.reduced_dim:
        print('explained_var_ratio', np.sum(be.named_steps['reducer'].explained_variance_ratio_))

    print('\nHyperparameters of the best estimator')
    print("------------------")
    print("Directly from the estimator:")
    best_params_keys = [k for k in param_grid.keys()]
    for kk in best_params_keys:
        print('%s : %.3f'%(kk, search.best_estimator_.get_params()[kk]))
    # Verify these best values by double-checking CV results df
    rowid = cv_df['mean_test_score'].argmax()
    bestrowdf = cv_df.iloc[rowid:rowid+1][
        search_param_list+ ['mean_train_score', 'mean_test_score']]
    print("As stored in the cv results table:")
    print(bestrowdf[search_param_list].T.to_string(header=False))

    print("\nPerformance summary")
    print("-------------------")
    print("ROC AUC for best hypers from K-fold CV, where 'test' is avg over heldout folds")
    print(bestrowdf[['mean_train_score', 'mean_test_score']].T.to_string(header=False))
    
    # Evaluate predictions of the ultimate model
    x_tr_MF = X_train
    ytrue_tr_M = y_train
    x_te_NF = X_test
    ytrue_te_N = y_test

    yproba1_tr_M = be.predict_proba(x_tr_MF)[:, 1]
    yproba1_te_N = be.predict_proba(x_te_NF)[:, 1]

    print("ROC AUC for final best estimator (after refit)")
    print('train : %.3f' % roc_auc_score(
        ytrue_tr_M, yproba1_tr_M))
    print('test : %.3f' % roc_auc_score(
        ytrue_te_N, yproba1_te_N))


