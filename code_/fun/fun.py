import numpy as np
import pandas as pd
from pyexplainer.pyexplainer import pyexplainer_pyexplainer
import warnings

warnings.filterwarnings("ignore")


def autoSpearman(x_data):
    res_autoSpearman = pyexplainer_pyexplainer.AutoSpearman(x_data, correlation_threshold=0.7,
                                                            correlation_method='spearman', VIF_threshold=5)
    features = res_autoSpearman.columns.values
    return features

def dataprocess(data, label, lt):
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
        codeData.drop(columns=lt, inplace=True)
        selected_cols = autoSpearman(codeData)
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    elif label == 'cs_':
        # 1.2.3 code-noSize_all-SNA(cs_)
        codeData = data.drop(columns=lt)
        selected_cols = autoSpearman(codeData)
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    elif label == 'codeSNA':
        # 1.2.4 all_code+SNA
        selected_cols = autoSpearman(data.drop(columns=['bug']))
        selected_cols = np.append(selected_cols, "bug")
        new_data = data.loc[:, selected_cols]
    return new_data


# 列出文件目录
'''
validation = "cross_validation"
for path, catalogue, name in os.walk("../res/rq3-res/" + validation + "/sk_esd_rank/"):
    print(name)
'''

# 判断列是否全为NAN值
def isNull(df):
    for cols in df.columns.values:
        # 删除全0或空列
        if (df[cols] == 0).all() or df[cols].isnull().all():
            df.drop(columns=cols, inplace=True)
    # print(df)
    return df