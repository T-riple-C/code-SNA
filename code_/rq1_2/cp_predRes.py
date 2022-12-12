# 模型性能-cross-project
import sys
sys.path.append("..")
from code_.fun import fun,models,config
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

predModel = ['rf', 'nb', 'xgb', 'lr','svm']
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
labels = ['all_code', 'code_nosize', 'cs_']
datasets = config.datasets
cols = config.cols
# res存储模型预测结果
res = pd.DataFrame(None, columns=['roc_auc', 'acc', 'mcc', 'metrics'])

for model in predModel:
    res.to_csv("../../res/rq1_2-res/cross_project/pred_res/" + model + "/predRes.csv", mode='a', sep=',', index=False)
    if model == 'rf':
        m = RandomForestClassifier()
    elif model == 'lr':
        m = LogisticRegression()
    elif model == 'xgb':
        m = xgboost.XGBClassifier()
    elif model == 'nb':
        m = GaussianNB()
    elif model == 'svm':
        m = SVC(probability=True)
    for label in labels:
        for i in range(0,len(datasets)):
            trainData = pd.DataFrame(None, columns=cols,dtype=np.float)
            for j in range(0,len(datasets)):
                if i != j:
                    # 若i不等于j,datasets[j]对应的数据应该都作为训练集   # 外层i-tr;内层j-te——LOOCV——1-训练集/其余项目-测试集
                    tr_data = pd.read_csv("../../data/SC/" + datasets[j])
                    tr_data['bug'] = np.where((tr_data['bugs'] != 0), 1, 0)  # 0-非defect；1-defect
                    tr_data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
                    tr_data.fillna(0, inplace=True)
                    trainData = trainData.append(tr_data, ignore_index=True)
            te_data = pd.read_csv("../../data/SC/" + datasets[i])
            te_data['bug'] = np.where((te_data['bugs'] != 0), 1, 0)
            te_data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
            te_data.fillna(0, inplace=True)

            trainLen = len(trainData)
            allData = trainData.append(te_data, ignore_index=True)
            allData = fun.dataprocess(allData, label, lt)
            trainData = allData.iloc[0:trainLen]
            testData = allData.iloc[trainLen:]
            print(allData.columns.values)

            X_train = trainData.drop(labels=['bug'], axis=1)
            y_train = trainData['bug'].astype(dtype='int32')
            X_test = testData.drop(labels=['bug'], axis=1)
            y_test = testData['bug'].astype(dtype='int32')

            del trainData
            oversampler = SMOTE(random_state=2, k_neighbors=2)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)

            m.fit(X_train, y_train)
            pred = m.predict_proba(X_test)
            roc_auc, acc, mcc, = models.evaluation(y_test, pred)
            tmp_res = pd.DataFrame({'roc_auc': [roc_auc], 'acc': [acc], 'mcc': [mcc], "metrics": [label]})
            res = res.append(tmp_res, ignore_index=True)
            # break
        # break
    res.to_csv("../../res/rq1_2-res/cross_project/pred_res/" + model + "/predRes.csv", mode='a', sep=',', index=None,
               header=False)
    print(model+" done.")