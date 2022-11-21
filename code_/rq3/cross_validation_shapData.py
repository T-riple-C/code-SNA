# shap所有项目的单解释样本(sk-esd输入)   +    特征总体重要性数据
import shap
from imblearn.under_sampling import RandomUnderSampler

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
# m = RandomForestClassifier()
tag = 't15'  # 只考虑t15
# lt = fun.to_lt(tag)
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
datasets = config.datasets  # 统一删除极不平衡、数据少的
col = config.cols
if 'bug' in col:
    col.remove('bug')
t_1 = pd.DataFrame(columns=col)
col.extend(['pro','method'])
t_sum = pd.DataFrame(columns=col)   #fsiSum的结果-pro,method

# shap_fsiSum 是所有项目数据汇总三种label得到的
# todo
# t_sum.to_csv('../../res/rq3-res/cross_validation/fsiSum/' + tag + '-shap_fsiSum.csv', mode='a', index=False)
for label in ['codeSNA']: # 'all_code', 'code_nosize', 'cs_', #todo
    for i in range(len(datasets)):
        print(datasets[i] + "-data**********************************")
        data_name = datasets[i]
        # todo
        # t_1.to_csv('../../res/rq3-res/cross_validation/shap_explainerSample/' + label + '/' + tag + '_' +
        #             data_name[:-11] + '.csv', mode='a', index=False)
        # 1.1 read CSV file
        data = pd.read_csv("../../data/SC/" + data_name)
        data['bug'] = np.where((data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
        data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
        data.fillna(0, inplace=True)
        print(data)
        new_data = fun.dataprocess(data, label, lt)
        new_data = new_data.sample(frac=1)  # 其中frac=1代表将打乱后的数据全部返回
        print(new_data.shape)
        # 1.3 split train & test
        # kf = KFold(n_splits=10)
        X = new_data.drop(labels=['bug'], axis=1)
        Y = new_data['bug']
        cnt = 0
        inx_len = 0
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=30)
        # 1.4 balance
        oversampler = SMOTE(random_state=2, k_neighbors=4)
        try:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
        except:
            print('balanced data error!!!!!')
        finally:
            pass

        # rf-随机森林模型

        m = RandomForestClassifier().fit(X_train, y_train)
        explainer = shap.TreeExplainer(m)
        shap_values_1 = explainer.shap_values(X_train)

        # shap_plot
        # todo
        # '''
        shap.summary_plot(shap_values_1[1], X_train, show=False, max_display=10)  # ——类蜂窝图
        plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
        plt.savefig("../../res/rq3-res/cross_validation/shap_plotBW/" + label + '/' + tag + '_' + datasets[i][:-11] + ".png")
        # plt.show()
        plt.close()
        shap.summary_plot(shap_values_1[1], X_train, plot_type="bar", show=False, max_display=10)
        plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
        plt.savefig("../../res/rq3-res/cross_validation/shap_plot/" + label + '/' + tag + '_' + datasets[i][:-11] + ".png")
        # plt.show()
        plt.close()
        # '''

        # todo
        # shap_data
        '''
        tt1 = pd.DataFrame(shap_values_1[0], columns=X_test.columns)
        tt2 = pd.DataFrame(shap_values_1[1], columns=X_test.columns)
        tmp_1 = abs(pd.concat([tt1, tt2], axis=0, ignore_index=True))
        '''
        # 保存单个样本的解释结果——skesd特征排名的输入
        '''
        t_1 = t_1.append(tmp_1)
        # print(t_1)
        t_1.to_csv('../../res/rq3-res/cross_validation/shap_explainerSample/' + label + '/' + tag + '_' + data_name[:-11] + '.csv', mode='a', index=False,header=False)
        t_1.drop(index=t_1.index, inplace=True)
        '''

        # 取单个解释样本的平均值，汇总——fsiSum
        '''
        tmp_mean = tmp_1.mean()
        tmp_mean = tmp_mean.to_frame()  # series
        tmp_meanT = pd.DataFrame(tmp_mean.values.T, columns=tmp_mean.index.values)
        tmp_meanT.insert(loc=0, column='method', value=label)
        tmp_meanT.insert(loc=0, column='pro', value=datasets[i][:-11])

        t_sum = t_sum.append(tmp_meanT)
        # break
    # print(t_sum)
    t_sum.to_csv('../../res/rq3-res/cross_validation/fsiSum/' + tag + '-shap_fsiSum.csv', mode='a', header=False, index=False)
    t_sum.drop(index=t_sum.index, inplace=True)
    '''
    # break
print("done.")
