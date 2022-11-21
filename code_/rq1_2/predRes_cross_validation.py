# 模型效果预测值
import sys
sys.path.append("..")
from code_.fun import config,fun,models
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")

model = 'svm'  # 'rf', 'nb','xgb','lr','svm'
# lt = fun.to_lt(tag) #to_lt里包含了bugs列，改后的代码不需要删除bugs列
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
dataset = config.datasets   # 删除数据极不平衡的几个数据集
print(dataset)
for id in range(1,17):
    # tag = 't'+str(id)
    tag = 't15' # 只t15的结果
    res = pd.DataFrame({'fname': ["fname"], 'roc_auc': ["roc_auc"], 'acc': ["acc"], 'mcc': ["mcc"], 'type': [tag]})
    res.to_csv("../../res/rq1_2-res/cross_validation/pred_res/"+model+"/"+tag+".csv", mode='a', sep=',', index=False, header=False)
    for label in ['all_code','code_nosize','cs_']:   # 'c_s_' 统一删除c_s_
        for i in range(len(dataset)):
            print(dataset[i] + "-data**********************************")
            data_name = dataset[i]
            # 1.1 read CSV file
            data = pd.read_csv("../../data/SC/" + data_name)
            data['bug'] = np.where((data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
            data = data.drop(columns=['Project','Version','name','bugs'])
            data.fillna(0, inplace=True)
            # print(data.columns.values)
            new_data = fun.dataprocess(data,label,lt)
            new_data = new_data.sample(frac=1)  # 其中frac=1代表将打乱后的数据全部返回
            print(new_data.shape)

            # 1.3 split train & test
            kf = KFold(n_splits=10)
            X = new_data.drop(labels=['bug'], axis=1)
            Y = new_data['bug']
            cnt = 0
            tmp_res = np.empty([10, 3], dtype=float)
            for train_index, test_index in kf.split(X):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
                y_train = Y.iloc[train_index]
                y_test = Y.iloc[test_index]
                # 1.4 balance
                if (np.sum(y_train == 1) <= np.sum(y_train == 0) / 2):  # 判断是否需要平衡
                    if (np.sum(y_train == 1) < 5):
                        oversampler = SMOTE(random_state=0, k_neighbors=1)
                    else:
                        oversampler = SMOTE(random_state=2, k_neighbors=3)
                elif (np.sum(y_train == 0) <= np.sum(y_train == 1) / 2):
                    if (np.sum(y_train == 0) < 5):
                        oversampler = SMOTE(random_state=0, k_neighbors=1)
                    else:
                        oversampler = SMOTE(random_state=2, k_neighbors=2)
                else:
                    oversampler = SMOTE(random_state=2, k_neighbors=4)
                X_train, y_train = oversampler.fit_resample(X_train, y_train)

                if model == 'rf':
                    res = models.run_randomForests(dataset[i][:-11], X_train, X_test, y_train, y_test,label)
                elif model == 'lr':
                    res = models.run_logistic(dataset[i][:-4], X_train,X_test,y_train,y_test,label)
                elif model == 'xgb':
                    res = models.run_XGB(dataset[i][:-4], X_train,X_test,y_train,y_test,label)
                elif model == 'nb':
                    res = models.run_NB(dataset[i][:-11], X_train,X_test,y_train,y_test,label)
                elif model == 'svm':
                    res = models.run_SVC(dataset[i][:-11], X_train, X_test, y_train, y_test, label)
                if (cnt < 10):
                    tmp_res[cnt] = res.values[0][1:-1]
                cnt = cnt + 1
            tmp = list(np.average(tmp_res, axis=0))
            tmp.insert(0, dataset[i][:-11])
            tmp.append(label)
            res.loc[0] = tmp
            res.to_csv("../../res/rq1_2-res/cross_validation/pred_res/"+model+"/"+tag+".csv", mode='a', sep=',', index=None, header=False)
        print("done.")
        # break
    break  # 只考虑t15才break外层循环
