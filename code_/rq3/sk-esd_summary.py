# 全部单解释样本——特征重要性排名——sk-esd——总和
from code_.fun import config
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri

pandas2ri.activate()
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# validation = "cross_validation"
validation = "cross_project_validation"
dataset = config.datasets  # 统一数据
all_cols = config.cols_skesd

sk = importr('ScottKnottESD')
tag = 't15'
labels = ['all_code', 'code_nosize', 'cs_', 'codeSNA']
idx = 0

df_new_1 = pd.DataFrame(columns=all_cols)
df_new_4 = pd.DataFrame(columns=all_cols)
rank_prob = pd.DataFrame(columns=all_cols)

for label in labels:
    for i in range(len(dataset)):
        data = pd.read_csv("../../res/rq3-res/"+validation+"/shap_explainerSample/" + label + "/" + tag + "_" + dataset[i][:-11] + ".csv")
        data.fillna(0, inplace=True)
        tmpCol = data.T.loc[(data.T != np.float32(0)).all(axis=1)]  # 找到不全为0的列
        data = data[tmpCol.index.values]
        robjects.globalenv['dataframe'] = data
        r_sk = sk.sk_esd(data)

        # '''
        # r_sk[4]对应特征重要性
        cols = list(r_sk[4].names[0])
        tmp_4 = list(r_sk[4][0:len(cols)])
        tmp_4.append(label)
        tmp_4.append(dataset[i][:-11])
        cols.append('method')
        cols.append('pro')
        ranking_4 = pd.DataFrame([tmp_4], columns=cols)
        for col in ranking_4.columns.values:
             df_new_4.loc[idx, col] = ranking_4[col].values[0]
        idx = idx + 1
        # '''

    # '''
        # r_sk[1]对应特征排名
        ranking_1 = pd.DataFrame([list(r_sk[1])], columns=r_sk[1].names)
        for col in ranking_1.columns.values:
            df_new_1.loc[dataset[i][:-11], col] = ranking_1[col].values[0]
    df_new_1.to_csv("../../res/rq3-res/"+validation+"/sk_esd_rank/" + label + "_" + tag + '-rank.csv')

    '''
    # 总结方法:统计每个度量出现在第一个rank的概率（比例）
    for tmp_1 in df_new_1.columns.values:
        if 1 in df_new_1[tmp_1].values:
            rank_prob.loc[0, tmp_1] = df_new_1[tmp_1].value_counts()[1]
    rank_prob = rank_prob / len(df_new_1.columns.values)
    
    for tmp1 in rank_prob.columns.values:
        if rank_prob[tmp1].isnull().all():
            rank_prob.drop(columns=tmp1, inplace=True)
    
    rank_prob = rank_prob.T.sort_values(axis=0, by=0, ascending=False)
    # print(rank_prob)
    plt.figure(figsize=(9, 7))
    plt.plot(rank_prob)
    x = range(0, len(rank_prob.index), 1)
    plt.xticks(x, rank_prob.index, color='blue', rotation=45)
    plt.ylabel('rank')
    plt.savefig("../../res/rq3-res/"+validation+"/sk_esd_rank/" + label + "_" + tag + "-rank1.png")  # 所有数据汇总的结果
    plt.close()
    # plt.show()
    '''
    df_new_1.drop(index=df_new_1.index, inplace=True)
    rank_prob.drop(index=rank_prob.index, inplace=True)
    # '''
    # break
print(df_new_4)
df_new_4.to_csv("../../res/rq3-res/"+validation+"/fsiSum/"+tag+'-sk-esd_fsiSum.csv',index=False)

