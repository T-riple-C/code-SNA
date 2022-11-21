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
# t_1 = pd.DataFrame(columns=col)
col.extend(['pro','method'])
# t_sum = pd.DataFrame(columns=col)   #fsiSum的结果-pro,method

# shap_fsiSum 是所有项目数据汇总三种label得到的
# t_sum.to_csv('../../res/rq3-res/cross_validation/fsiSum/' + tag + '-shap_fsiSum.csv', mode='a', index=False)
for label in ['cs_']:  # 'all_code', 'code_nosize', 'cs_'
    for i in range(len(datasets)):
        print(datasets[i] + "-data**********************************")
        data_name = datasets[i]
        # explainerSample 是根据项目（解释样本）得到的;每个项目需要一个新的csv
        # t_1.to_csv('../../res/rq3-res/cross_validation/shap_explainerSample/' + label + '/' + tag + '_' +
        #             data_name[:-11] + '.csv', mode='a', index=False)
        # 1.1 read CSV file
        data = pd.read_csv("../../data/SC/" + data_name)
        data['bug'] = np.where((data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
        data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
        data.fillna(0, inplace=True)
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
        # print(shap_values_1)
        # shap_plot
        # '''
        # shap_value特征重要性——beewarm
        # shap.summary_plot(shap_values_1[1], X_train, show=False, max_display=10)  # ——类蜂窝图
        # shap.summary_plot(shap_values_1[1], X_train, plot_type="bar", show=False, max_display=10)


        # shap_interaction_values
        # 单个特征对预测值的影响

        # 根据实验前部分结果 Betweenness，pWeakC(out)，avg(cc)，2StepP(out)，lcom，cam，lcom3
        # 从单特征图2stepP(out)越大，相对SHAP值小，这个特征与2stepReach的关系？
        # print(X_train.columns.values)
        '''
        for fs in X_train.columns.values:
            shap.dependence_plot(fs, shap_values_1[0], X_train, show=False, interaction_index=None)

            plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
            plt.savefig("../../res/rq3-res/cross_validation/shap_dependence/" + label + '/' + datasets[i][:-11] + '_' +
                        fs + ".png")  # 可以保存图片
            # plt.show()
            plt.close()
        '''
        FS = ['lcom3', 'lcom', 'cam', 'dam', 'avg(cc)']
        # '''
        # cs_
        # 一个特征如何和另一个特征交互影响预测结果的
        # betweenness、2stepP(in,out,un)、pWeakC(in,out,un)、nbroke(in,out,un),hierarchy

        for item in FS:
            if item in X_train.columns.values:
                try:
                    shap.dependence_plot(item, shap_values_1[0], X_train, show=False, interaction_index='pWeakC(out)')
                    plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                    plt.savefig(
                        "../../res/rq3-res/cross_validation/shap_2dependence/" + label + '/pWeakC_out/' + datasets[i][:-11]
                        + '_' + item + ".png")  # 可以保存图片
                    # plt.show()
                    plt.close()
                except:
                    print("fs not in cols-error!")
                finally:
                    pass
        # '''
        '''
        # all_code
        # size:"amc", "loc", "npm", "noc", "moa
        for item in FS:
            if item in X_train.columns.values:
                try:
                    shap.dependence_plot(item, shap_values_1[0], X_train, show=False, interaction_index='moa')
                    plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                    plt.savefig(
                        "../../res/rq3-res/cross_validation/shap_2dependence/" + label + '/moa/' + datasets[i][:-11]
                        + '_' + item + ".png")  # 可以保存图片
                    # plt.show()
                    plt.close()
                except:
                    print("fs not in cols-error!")
                finally:
                    pass
        '''
        '''
        plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
        # plt.savefig("../../res/rq3-res/cross_validation/shap_plotBW/" + label + '/' + tag + '_' + datasets[i][:-11] + ".png")  # 可以保存图片
        plt.show()
        plt.close()
        '''

        '''
        # shap_data
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
    # break
print("done.")
