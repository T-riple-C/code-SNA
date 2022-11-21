#todo
# 各个功能分开；数据集等-配置-config
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.data import _is_using_pandas
from pyexplainer.pyexplainer import pyexplainer_pyexplainer

import warnings

warnings.filterwarnings("ignore")


def autoSpearman(x_data):
    res_autoSpearman = pyexplainer_pyexplainer.AutoSpearman(x_data, correlation_threshold=0.7,
                                                            correlation_method='spearman', VIF_threshold=5)
    features = res_autoSpearman.columns.values
    # print(features)
    return features


def to_lt(tag):
    tags = {
        't1': ["amc", "loc", "bug"],
        't2': ["amc", "loc", "wmc", "bug"],
        't3': ["amc", "loc", "noc", "bug"],
        't4': ["amc", "loc", "npm", "bug"],
        't5': ["amc", "loc", "moa", "bug"],
        't6': ["amc", "loc", "wmc", "noc", "bug"],
        't7': ["amc", "loc", "wmc", "npm", "bug"],
        't8': ["amc", "loc", "wmc", "moa", "bug"],
        't9': ["amc", "loc", "noc", "npm", "bug"],
        't10': ["amc", "loc", "noc", "moa", "bug"],
        't11': ["amc", "loc", "npm", "moa", "bug"],
        't12': ["amc", "loc", "wmc", "noc", "npm", "bug"],
        't13': ["amc", "loc", "wmc", "noc", "moa", "bug"],
        't14': ["amc", "loc", "wmc", "npm", "moa", "bug"],
        't15': ["amc", "loc", "npm", "noc", "moa", "bug"],
        't16': ["amc", "loc", "wmc", "noc", "moa", "npm", "bug"],
    }
    return tags.get(tag)


def dataprocess(data, label, lt):
    # print('dataprocess1:',data.columns.values)
    if label == 'all_code':
        # 1.2.1 all code
        codeData = data.iloc[:, 63:]
        codeData.drop(columns=["bug"], inplace=True)
        selected_cols = autoSpearman(codeData)
        selected_cols = np.append(selected_cols, 'bug')
        new_data = data.loc[:, selected_cols]
    elif label == 'code_nosize':
        # 1.2.2 code-noSize
        codeData = data.iloc[:, 63:]
        codeData['bug'] = data['bug']
        # print(codeData.columns.values)
        codeData.drop(columns=lt, inplace=True)
        selected_cols = autoSpearman(codeData)
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    elif label == 'cs_':
        # 1.2.4 code-noSize_all-SNA(cs_auto)
        codeData = data.drop(columns=lt)
        selected_cols = autoSpearman(codeData)
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    # new_data = new_data.sample(frac=1)
    elif label == 'codeSNA':
        selected_cols = autoSpearman(data.drop(columns=['bug']))
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    return new_data

