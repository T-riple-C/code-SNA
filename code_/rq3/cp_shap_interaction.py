# 模型效果预测值——cross_project
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
m = RandomForestClassifier()
tag = 't15'
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]

labels = ["cs_"]  # , 'cs_', 'code_nosize',
datasets = config.datasets
col = config.cols
if 'bug' in col:
    col.remove('bug')
data_col = col.copy()
# t_1 = pd.DataFrame(columns=col)
col.extend(['pro','method'])
# t_sum = pd.DataFrame(columns=col)   #fsiSum的结果-pro,method

# t_sum.to_csv('../../res/rq3-res/cross_project_validation/fsiSum/' + tag + '-shap_fsiSum.csv', mode='a', index=False)

# 从结果导向看，对每种方法-统计所有训练集和测试集划分情况所构建预测模型的预测结果，再对三种方法进行比较，貌似也合乎逻辑？
for label in labels:
    # 双循环,不用k-fold
    for i in range(0, len(datasets)):
        trainData = pd.DataFrame(None, columns=data_col, dtype=np.float)
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
        # Dataframe.info()  X_train.info()可以查看各数据类型

        # 模型预测
        m.fit(X_train, y_train)

        # shap——（仅-随机森林——用TreeExplainer）
        explainer = shap.TreeExplainer(m)
        shap_values_1 = explainer.shap_values(X_test)
        # test进行测试
        # print(shap_values_1)
        # '''
        # shap_plot
        # shap_values_summary_plot
        # shap.summary_plot(shap_values_1[1], X_test, show=False, max_display=10)  # ——类蜂窝图
        # shap.summary_plot(shap_values_1[1], X_train, plot_type="bar", show=False, max_display=10)

        # shap_interaction_values_dependence_plot
        '''
        # 单个特征对预测结果影响-每个数据集(size+SNA)-参与预测的度量特征-对预测结果影响的dependence图
        for fs in X_test.columns.values:
            shap.dependence_plot(fs, shap_values_1[0], X_test, show=False, interaction_index=None)
            plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
            plt.savefig("../../res/rq3-res/cross_project_validation/shap_dependence/" + label + '/' + datasets[i][:-11]
                        + fs + ".png")  # 可以保存图片
            # plt.show()
            plt.close()
        '''
        # 两个特征—一个特征对另一个特征交互作用对预测的影响
        FS = ['lcom3', 'lcom', 'cam', 'dam', 'avg(cc)']
        # '''
        # cs_ 
        # SNA [betweenness、pWeakC(in,out,un)、nBroke(in,out,un),hierarchy、2stepP(in,out,un)]
        for item in FS:
            if item in X_test.columns.values:
                try:
                    shap.dependence_plot(item, shap_values_1[0], X_test, show=False, interaction_index='nBroke(out)')
                    plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                    plt.savefig(
                        "../../res/rq3-res/cross_project_validation/shap_2dependence/" + label + '/nBroke(out)/' + datasets[i][:-11]
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
        # size = ["amc", "loc", "npm", "noc", "moa"]
        for item in FS:
            if item in X_test.columns.values:
                try:
                    shap.dependence_plot(item, shap_values_1[0], X_test, show=False, interaction_index='noc')
                    plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                    plt.savefig(
                        "../../res/rq3-res/cross_project_validation/shap_2dependence/" + label + '/noc/' + datasets[i][:-11]
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
        # plt.savefig("../../res/rq3-res/cross_project_validation/shap_plotBW/" + label + '/' + tag + '_' + datasets[i][:-11] + ".png")  # 可以保存图片
        plt.show()
        plt.close()
        '''
        '''
        # shap_data

        tt1 = pd.DataFrame(shap_values_1[0], columns=X_test.columns)
        tt2 = pd.DataFrame(shap_values_1[1], columns=X_test.columns)
        # 保存单个样本的解释结果
        tmp_1 = abs(pd.concat([tt1, tt2], axis=0, ignore_index=True))
        
        t_1 = t_1.append(tmp_1)
        # print(t_1)
        
        t_1.to_csv('../../res/rq3-res/cross_project_validation/shap_explainerSample/' + label + '/' + tag + '_' + datasets[i][:-11] + '.csv',
                   index=False)
        '''
        '''
        # 取单个解释样本的平均值，汇总——fsiSum
        tmp_mean = tmp_1.mean()
        tmp_mean = tmp_mean.to_frame()  # series
        tmp_meanT = pd.DataFrame(tmp_mean.values.T, columns=tmp_mean.index.values)
        tmp_meanT.insert(loc=0, column='method', value=label) # 由于tmp1是tmp从series转置来的，其loc的范围0-11
        tmp_meanT.insert(loc=0, column='pro', value=datasets[i][:-11])
        # i对应的是30个数据项目; # 每次i层循环结束，tt对应有30条数据；
        t_sum = t_sum.append(tmp_meanT)
        # print(t_sum)
        # break
    t_1.drop(index=t_1.index, inplace=True)
    '''
        # break
    # break
# t_sum.to_csv('../../res/rq3-res/cross_project_validation/fsiSum/' + tag + '-shap_fsiSum.csv', mode='a', header=False, index=False)
print("done.")
