# cross-project
# SHAP dependency plot
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

lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
datasets = config.datasets

for label in ["all_code", "cs_"]:
    for i in range(0, len(datasets)):
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
        oversampler = SMOTE(random_state=42, k_neighbors=2)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        # rf-模型训练/SHAP计算
        m = RandomForestClassifier().fit(X_train, y_train)

        explainer = shap.TreeExplainer(m)
        shap_values = explainer.shap_values(X_train)
        # print(shap_values)

        # SHAP dependency plot
        # -------------单个特征对预测结果影响
        
        for fs in X_train.columns.values:
            shap.dependence_plot(fs, shap_values[0], X_train, show=False, interaction_index=None)
            plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
            plt.savefig("../../res/rq3-res/cross_project/shap_dependency/" + label + '/' + datasets[i][:-11] + '_'
                        + fs + ".png")
            # plt.show()
            plt.close()

        # -------------两个特征交互影响
        codeNosize_FS = ['lcom3', 'lcom', 'cam', 'dam', 'avg(cc)', 'ca']
        SNA_FS = ['Betweenness', '2StepP(out)', '2StepR(un)', 'pWeakC(out)', 'nBroke(out)', 'nBroke(un)', 'Eigenvector']
        size_FS = ["amc", "loc", "npm", "noc", "moa"]

        for item in codeNosize_FS:
            if item in X_train.columns.values:
                # SNA度量与其他非code度量特征交互
                for snaFS in SNA_FS:
                    try:
                        shap.dependence_plot(item, shap_values[0], X_train, show=False, interaction_index=snaFS)
                        plt.tight_layout()  # 充分显示坐标
                        path1 = "../../res/rq3-res/cross_project/shap_dependency2/" + label + '/'+snaFS+'/' +\
                                datasets[i][:-11] + '_' + item + ".png"
                        plt.savefig(path1)
                        # plt.show()
                        plt.close()
                    except:
                        print("sna,fs not in cols-error!")
                    finally:
                        pass
                # 体积度量与其他非体积code度量特征交互
                for sizeFS in size_FS:
                    try:
                        shap.dependence_plot(item, shap_values[0], X_train, show=False, interaction_index=sizeFS)
                        plt.tight_layout()
                        path2 = "../../res/rq3-res/cross_project/shap_dependency2/" + label + '/'+sizeFS+'/' + \
                                datasets[i][:-11] + '_' + item + ".png"
                        plt.savefig(path2)
                        # plt.show()
                        plt.close()
                    except:
                        print("size,fs not in cols-error!")
                    finally:
                        pass
        print(i, "--plot done.")
        # break
    # break
print("done.")