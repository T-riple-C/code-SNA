# cross_project
# shap所有项目的单解释样本(sk-esd输入)   +    特征总体重要性数据

import sys
import shap

sys.path.append("../..")
from code_.fun import fun, config
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings("ignore")

model = 'rf'
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
labels = ['all_code', 'code_nosize', 'cs_']
datasets = config.datasets
col = config.cols.copy()
col.extend(['pro','method'])
t_sum = pd.DataFrame(columns=col)

t_sum.to_csv('../../res/rq3-res/cross_project/fsiSum/shap_fsiSum.csv', mode='a', index=False)
for label in labels:
    for i in range(0, len(datasets)):
        print(i,", current test:",datasets[i])
        trainData = pd.DataFrame(None, columns=config.cols, dtype=np.float)
        for j in range(0, len(datasets)):
            if i != j:
                # 若i不等于j,datasets[j]对应的数据应该都作为训练集   # 外层i-tr;内层j-te——LOOCV——1-训练集/其余项目-测试集
                tr_data = pd.read_csv("../../data/SC/" + datasets[j])
                tr_data['bug'] = np.where((tr_data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
                tr_data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
                tr_data.fillna(0, inplace=True)
                trainData = trainData.append(tr_data, ignore_index=True)
        te_data = pd.read_csv("../../data/SC/" + datasets[i])
        te_data['bug'] = np.where((te_data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
        te_data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
        te_data.fillna(0, inplace=True)

        trainLen = len(trainData)
        allData = trainData.append(te_data, ignore_index=True)  # testData 只选一个项目作为测试集
        # print(allData.columns.values)
        allData = fun.dataprocess(allData, label, lt)
        trainData = allData.iloc[0:trainLen]
        testData = allData.iloc[trainLen:]
        # print(trainData,testData)

        X_train = trainData.drop(columns=['bug'])
        y_train = trainData['bug'].astype(dtype='int32')  # 0、1缺陷标签，object-int32
        X_test = testData.drop(columns=['bug'])  # labels=['bug'], axis=1
        y_test = testData['bug'].astype(dtype='int32')

        del trainData
        oversampler = SMOTE(random_state=2, k_neighbors=2)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        # 模型预测
        m = RandomForestClassifier().fit(X_train, y_train)
        explainer = shap.TreeExplainer(m)

        shap_values = explainer.shap_values(X_train)
        tmpSHAP = pd.DataFrame(columns=config.cols)
        # 假设考虑的是shap_values[0]的结果
        # shap_plot——保存每次训练集/测试集对应的蜂窝图
        # '''
        shap.summary_plot(shap_values[0], X_train, show=False, max_display=10)  # ——类蜂窝图
        plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
        plt.savefig("../../res/rq3-res/cross_project/shap_plot/" + label + '/' + datasets[i][:-11] + ".png")
        # plt.show()
        plt.close()
        # '''

        # shap_data——保存单个样本的解释结果
        tt1 = pd.DataFrame(shap_values[0], columns=X_train.columns)
        tmpSHAP = pd.concat([tmpSHAP, tt1], axis=0)
        tmpSHAP.to_csv('../../res/rq3-res/cross_project/shap_explainerSample/' + label + '/' + datasets[i][:-11] + '.csv',
                   index=False)

        # 每个数据项目——汇总各特征-shap_value平均值
        shap_valueMean = abs(tmpSHAP).mean()
        shap_valueMean['pro'] = datasets[i][:-11]
        shap_valueMean['method'] = label
        shap_valueMean = shap_valueMean.to_frame()
        shap_valueMeanT = pd.DataFrame(shap_valueMean.values.T, columns=shap_valueMean.index.values)
        # print(shap_valueMeanT)
        t_sum = t_sum.append(shap_valueMeanT)
    t_sum.to_csv('../../res/rq3-res/cross_project/fsiSum/shap_fsiSum.csv', mode='a', header=False, index=False)
    t_sum.drop(index=t_sum.index, inplace=True)
print("done.")
