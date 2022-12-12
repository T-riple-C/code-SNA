# 模型封装
import pandas as pd
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

def evaluation(y, pred):
    try:
        roc_auc = format(roc_auc_score(y, pred[:, 1]))
    except ValueError:
        roc_auc = None
    acc = format(accuracy_score(y, pred[:, 1].round()))
    mcc = format(matthews_corrcoef(y, pred[:, 1].round()))
    return roc_auc, acc, mcc


def run_randomForests(fname, X_train, X_test, y_train, y_test, label):
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
                                max_depth=70, bootstrap=False)
    rf.fit(X_train, y_train)
    pred = rf.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res


def run_logistic(fname, X_train, X_test, y_train, y_test, label):
    logit = LogisticRegression(C=0.001, penalty='l2')
    logit.fit(X_train, y_train)
    pred = logit.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res


def run_ADA(fname, X_train, X_test, y_train, y_test, label):
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    pred = ada.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res


def run_DT(fname, X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    pred = dt.predict_proba(X_test)
    roc_auc_tt, acc_tt, f1_tt, p_tt, r_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'f1': [f1_tt], 'p': [p_tt],
                        'r': [r_tt], 'mcc': [mcc_tt]})
    return res


def run_XGB(fname, X_train, X_test, y_train, y_test, label):
    xgb = xgboost.XGBClassifier(learning_rate=0.01, max_depth=6, min_child_weight=4, n_estimators=80, subsample=0.7)
    xgb.fit(X_train, y_train)
    pred = xgb.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res


def run_NB(fname, X_train, X_test, y_train, y_test, label):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    pred = nb.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res


def run_SVC(fname, X_train, X_test, y_train, y_test, label):
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)
    pred = svm.predict_proba(X_test)
    roc_auc_tt, acc_tt, mcc_tt = evaluation(y_test, pred)
    res = pd.DataFrame({'fname': [fname], 'roc_auc': [roc_auc_tt], 'acc': [acc_tt], 'mcc': [mcc_tt], 'type': [label]})
    return res
