# 模型效果预测值——cross_project
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

# todo
# 去除数据集-数据不平衡——ant_1.3:57,synapse_1.0:72,jEdit_4.3:80,log4j_1.2.1:80,pbeans_1.0:80,pbeans_2.0:80,xalan_2.7.0:80
# 设使用LOOCV：每次选择一个项目作为 test-测试集，其余项目全部作为训练集；     双循环-确定测试集，并拼接训练集数据
# 需要确保训练集和测试集在进行autospearman后的度量集是一致的:上述数据准备完成后，需要将测试集和训练集拼接，在三种方法[all_code,code_nosize,cs_]中使用autospearman，保证度量子集的一致性；可以根据数据长度划分之前确定的测试集和训练集
# 再划分x,y的数据，训练

model = 'rf'
if model == 'rf':
    m = RandomForestClassifier()
elif model == 'lr':
    m = LogisticRegression()    #在cross-project validation的结果太尴尬了！
elif model == 'xgb':
    m = xgboost.XGBClassifier()
elif model == 'nb':
    m = GaussianNB()
elif model == 'svm':
    m = SVC(probability=True)

tag = 't15'
lt = ["amc", "loc", "npm", "noc", "moa", "bug"]  # wmc -多篇文章将其定性为复杂性度量，并且删除该度量对实验效果并没有提升
labels = ['all_code', 'code_nosize', 'cs_', ]
datasets = config.datasets
cols = config.cols
res = pd.DataFrame(None, columns=['roc_auc', 'acc', 'mcc', tag])    # 模型预测结果
# header
res.to_csv("../../res/rq1_2-res/cross_project_validation/pred_res/" + model + "/" + tag + ".csv", mode='a', sep=',', index=False)
# 从结果导向看，对每种方法-统计所有训练集和测试集划分情况所构建预测模型的预测结果，再对三种方法进行比较，貌似也合乎逻辑？
for label in labels:
    # 双循环,不用k-fold
    for i in range(0,len(datasets)):
        trainData = pd.DataFrame(None, columns=cols,dtype=np.float)
        for j in range(0,len(datasets)):
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
        allData = fun.dataprocess(allData, label, lt)
        trainData = allData.iloc[0:trainLen]
        testData = allData.iloc[trainLen:]
        print(allData.columns.values)

        X_train = trainData.drop(labels=['bug'], axis=1)
        y_train = trainData['bug'].astype(dtype='int32')  # 0、1缺陷标签，object-int32
        X_test = testData.drop(labels=['bug'], axis=1)
        y_test = testData['bug'].astype(dtype='int32')

        del trainData  # ,testData, allData
        oversampler = SMOTE(random_state=2, k_neighbors=2)
        X_train, y_train = oversampler.fit_resample(X_train, y_train)
        # Dataframe.info()  X_train.info()可以查看各数据类型

        #
        m.fit(X_train, y_train)
        pred = m.predict_proba(X_test)
        roc_auc, acc, mcc, = models.evaluation(y_test, pred)  # 因为cross-project 没有fname这个参数值，所以不用models里的模型
        tmp_res = pd.DataFrame({'roc_auc': [roc_auc], 'acc': [acc], 'mcc': [mcc], tag: [label]})
        res = res.append(tmp_res, ignore_index=True)
        break
    break
    # 共计30个项目数据，每次选择一个作为测试集，[all-code,code_nosize,cs_] 每个方法对应30行res结果，
res.to_csv("../../res/rq1_2-res/cross_project_validation/pred_res/" + model + "/" + tag + ".csv", mode='a', sep=',', index=None,
           header=False)
print("done.")