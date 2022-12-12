# within-project
# SHAP dependency plot
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

lt = ["amc", "loc", "npm", "noc", "moa", "bug"]
datasets = config.datasets

for label in ['all_code','cs_']:
    for i in range(len(datasets)):
        print(datasets[i] + "-data**********************************")
        data_name = datasets[i]

        # 1.1 read CSV file
        data = pd.read_csv("../../data/SC/" + data_name)
        data['bug'] = np.where((data['bugs'] != 0), 1, 0)  # 0-label没有缺陷；1-label有缺陷
        data.drop(columns=['Project', 'Version', 'name', 'bugs'], inplace=True)
        data.fillna(0, inplace=True)
        # 1.2 data-process
        new_data = fun.dataprocess(data, label, lt)
        new_data = new_data.sample(frac=1)  # 其中frac=1代表将打乱后的数据全部返回
        print(new_data.shape)

        # 1.3 split train & test
        X = new_data.drop(labels=['bug'], axis=1)
        Y = new_data['bug']
        cnt = 0
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=30)

        # 1.4 balance
        oversampler = SMOTE(random_state=42, k_neighbors=4)
        try:
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
        except:
            print('balanced data error!!!!!')
        finally:
            pass

        # 1.5 rf-模型训练/SHAP计算
        m = RandomForestClassifier().fit(X_train, y_train)
        explainer = shap.TreeExplainer(m)
        shap_values = explainer.shap_values(X_train)
        # print(shap_values)

        # SHAP dependency plot
        # ----------单个特征对预测值的影响
        # '''
        for fs in X_train.columns.values:
            shap.dependence_plot(fs, shap_values[0], X_train, show=False, interaction_index=None)
            plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
            plt.savefig("../../res/rq3-res/within_project/shap_dependency/" + label + '/' + datasets[i][:-11] + '_' +
                        fs + ".png")
            # plt.show()
            plt.close()
        # '''

        # -----------两个特征交互影响
        codeNosize_FS = ['lcom3', 'lcom', 'cam', 'dam', 'avg(cc)', 'ca']
        SNA_FS = ['Betweenness', '2StepP(out)', '2StepR(un)', 'pWeakC(out)', 'nBroke(out)', 'nBroke(un)', 'Eigenvector']
        size_FS = ["amc", "loc", "npm", "noc", "moa"]
        # '''
        for item in codeNosize_FS:
            if item in X_train.columns.values:
                # SNA度量与其他非code度量特征交互
                for snaFS in SNA_FS:
                    try:
                        shap.dependence_plot(item, shap_values[0], X_train, show=False, interaction_index=snaFS)
                        plt.tight_layout()  # 充分显示坐标
                        path1 = "../../res/rq3-res/within_project/shap_dependency2/" + label + '/'+snaFS+'/' + \
                                datasets[i][:-11] + '_' + item + ".png"
                        plt.savefig(path1)
                        # plt.show()
                        plt.close()
                    except:
                        print("sna, fs not in cols-error!")
                    finally:
                        pass
                # 体积度量与其他非体积code度量特征交互
                for sizeFS in size_FS:
                    try:
                        shap.dependence_plot(item, shap_values[0], X_train, show=False, interaction_index=sizeFS)
                        plt.tight_layout()  # 让坐标充分显示，如果没有这一行，坐标可能显示不全
                        path2 = "../../res/rq3-res/within_project/shap_dependency2/" + label + '/' + sizeFS + '/' +\
                                datasets[i][:-11] + '_' + item + ".png"
                        plt.savefig(path2)
                        # plt.show()
                        plt.close()
                    except:
                        print("size, fs not in cols-error!")
                    finally:
                        pass
        # '''
        # break
    # break
print("done.")