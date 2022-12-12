# 特征重要性平均值-单个实例-数据
from code_.fun import config, fun
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

validation = ["within_project","cross_project"]
labels = ['all_code', 'cs_']
dataset = config.datasets  # 所有数据
all_cols = config.cols_skesd

def sk_esdRank(dataset,v,label):
    for i in range(len(dataset)):
        # print(dataset[i])
        tmpdata = pd.read_csv(
            "../../res/rq3-res/" + v + "/shap_explainerSample/" + label + "/" + dataset[i][:-11] + ".csv")
        data = fun.isNull(abs(tmpdata))
        # print(data)
        data.fillna(0, inplace=True)
        robjects.globalenv['dataframe'] = data
        r_sk = sk.sk_esd(data)

        # r_sk[1]对应特征排名
        ranking_1 = pd.DataFrame([list(r_sk[1])], columns=r_sk[1].names)
        for col in ranking_1.columns.values:
            df_new.loc[dataset[i][:-11], col] = ranking_1[col].values[0]
    # print(df_new)
    # df_new.to_csv("../../res/rq3-res/" + v + "/sk_esd/" + label + '-rank.csv')
    # df_new.drop(index=df_new.index)
    return df_new

def shap(dataset,v,label,rankData):
    for i in range(len(dataset)):
        print('shap-----------------', v, label, dataset[i])
        tmpdata = pd.read_csv(
            "../../res/rq3-res/" + v + "/shap_explainerSample/" + label + "/" + dataset[i][:-11] + ".csv")
        data = fun.isNull(abs(tmpdata)).mean().sort_values(ascending=False)
        singleRankData = rankData.loc[dataset[i][:-11],:]
        singleRankData.rename(index={"avg.cc.": "avg(cc)"},inplace=True)         # avg(cc)  avg.cc.
        df = pd.concat([data, singleRankData], axis=1, ignore_index=False)
        df.dropna(inplace=True)
        df.columns = ['SHAP_value','SK-ESD']
        # print(df)
        df.to_csv("../../res/rq3-res/" + v + "/fsiSingle/" + label + '/' +dataset[i][:-11]+'.csv')
        # break

if __name__=='__main__':
    sk = importr('ScottKnottESD')
    df_new = pd.DataFrame(columns=all_cols)
    for v in validation:
        for label in labels:
            # sk-esd排名
            rankData = sk_esdRank(dataset,v,label)

            # shap_mean
            shap(dataset,v,label,rankData)
            # break
        # break