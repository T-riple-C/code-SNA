# within-project
# shap所有项目的单解释样本(sk-esd输入)   +   特征总体重要性数据
import shap
import sys
from matplotlib import pyplot as plt
sys.path.append("..")
from code_.fun import fun, config
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

model = 'rf'

lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
datasets = config.datasets  # 统一删除极不平衡、数据少的
col = config.cols.copy()
col.extend(['pro', 'method'])
t_sum = pd.DataFrame(columns=col)  # fsiSum的结果-pro,method

t_sum.to_csv('../../res/rq3-res/within_project/fsiSum/shap_fsiSum.csv', mode='a', index=False)
for label in ['all_code', 'code_nosize', 'cs_']:
    for i in range(len(datasets)):
        print(datasets[i] + "-data**********************************")
        data_name = datasets[i]

        # 1.1 read CSV file
        data = pd.read_csv("../../data/SC/" + data_name)
        data['bug'] = np.where((data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
        data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
        data.fillna(0, inplace=True)
        # print(data)
        new_data = fun.dataprocess(data, label, lt)
        new_data = new_data.sample(frac=1)  # 其中frac=1代表将打乱后的数据全部返回
        print(new_data.shape)
        # 1.3 split train & test
        X = new_data.drop(labels=['bug'], axis=1)
        Y = new_data['bug']
        cnt = 0
        inx_len = 0
        kf = KFold(n_splits=10, shuffle=True)
        # 设变量tmpSHAP用来保存十折后的结果
        tmpSHAP = pd.DataFrame(columns=config.cols)
        tmpSHAP.to_csv('../../res/rq3-res/within_project/shap_explainerSample/' + label + '/' +
                       data_name[:-11] + '.csv', mode='a', index=False)
        for train_index, test_index in kf.split(X):
            cnt = cnt + 1
            print(cnt)
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
            y_train = Y.iloc[train_index]
            y_test = Y.iloc[test_index]

            # 1.4 balance
            oversampler = SMOTE(random_state=42, k_neighbors=4)
            try:
                X_train, y_train = oversampler.fit_resample(X_train, y_train)
            except:
                print('balanced data error!!!!!')
            finally:
                pass

            # rf-随机森林模型
            m = RandomForestClassifier().fit(X_train, y_train)
            explainer = shap.TreeExplainer(m)
            shap_values = explainer.shap_values(X_train)

            # 假设考虑的是shap_values[0]的结果
            # shap_plot——只绘制最后一次的结果
            if cnt == 10:
                shap.summary_plot(shap_values[0], X_train, show=False, max_display=10)  # ——类蜂窝图
                plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                plt.savefig("../../res/rq3-res/within_project/shap_plot/" + label + '/' + datasets[i][:-11] + ".png")
                plt.show()
                plt.close()

            tt1 = pd.DataFrame(shap_values[0], columns=X_train.columns)
            tmpSHAP = pd.concat([tmpSHAP, tt1], axis=0)
        # 每条样本数据——十折交叉验证——由于oversample的结果不一致——必须全部保存
        tmpSHAP.to_csv('../../res/rq3-res/within_project/shap_explainerSample/' + label + '/' +
                       data_name[:-11] + '.csv',mode='a', index=False, header=False)

        # 每个数据项目——汇总各特征-shap_value平均值
        shap_valueMean = abs(tmpSHAP).mean()
        shap_valueMean['pro'] = datasets[i][:-11]
        shap_valueMean['method'] = label
        shap_valueMean = shap_valueMean.to_frame()
        shap_valueMeanT = pd.DataFrame(shap_valueMean.values.T, columns=shap_valueMean.index.values)
        # print(shap_valueMeanT)
        t_sum = t_sum.append(shap_valueMeanT)
    # print(t_sum)
    t_sum.to_csv('../../res/rq3-res/within_project/fsiSum/shap_fsiSum.csv', mode='a', header=False,
                 index=False)
    t_sum.drop(index=t_sum.index, inplace=True)
print("done.")
