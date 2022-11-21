#todo:目的，统一cross-validation和cross-project validation的代码
# sk-esd 模型性能__均值
# 两个函数：
# 1.sk_esdInput用于修改模型预测csv结果结构，便于使用skesd计算
# 2.res_img:将skesd——模型性能结果和排名分组结果画到一个图中

import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd
from matplotlib import pyplot as plt


def sk_esdInput(df,model,tag,validation):
    # 参数model和validation的目的变成确定使用的是哪种交叉验证结果文件
    cols = ['all_code', 'code_nosize', 'cs_']
    cols = cols.append(tag)
    df_new = pd.DataFrame(columns=[cols])
    items = ['roc_auc','acc','mcc']
    all_code = df[df[tag] == 'all_code']
    code_nosize = df[df[tag] == 'code_nosize']
    cs_ = df[df[tag] == 'cs_']
    allCode = []
    codeNosize = []
    CS = []
    itm = []
    for item in items:
        allCode.extend(all_code[item].values)
        codeNosize.extend(code_nosize[item].values)
        CS.extend(cs_[item].values)
        itm.extend(item for i in range(len(all_code)))
    df_new['all_code'] = allCode
    df_new['code_nosize'] = codeNosize
    df_new['cs_'] = CS
    df_new[tag] = itm
    # print(df_new)
    return df_new

def res_img(df,measure,tag,model,validation):
    robjects.globalenv['dataframe'] = df.iloc[:, 1:-1]
    r_sk1 = sk.sk_esd(df.iloc[:, 1:-1])

    # print(r_sk1[4])  # mean mean-s mean+s
    method = list(r_sk1[4].names[0])  # ['all_code','code_nosize','c_s_','cs_']
    m_value = list(r_sk1[4])  # mean*4, mean-s*4, mean+s*4
    df = pd.DataFrame(columns=[measure, tag])
    for i in range(0, len(method)):     # i method的index
        for j in range(0, 3):       # i+j*4_m_value的索引
            tmp = [[m_value[i + j * 3], method[i]]]
            df = df.append(pd.DataFrame(tmp, columns=[measure, tag]), ignore_index=True)
    rank = {}
    # print(r_sk1[1])
    for k in range(0, len(list(r_sk1[1].names))):
        rank[list(r_sk1[1].names)[k]] = list(r_sk1[1])[k]
    x = {'cs_': 3, 'code_nosize': 2, 'all_code': 1}
    for label in list(r_sk1[1].names):
        if label == 'all_code':
            # print(df[df[tag] == label]) # mean mean-s mean+s
            ac = df[df[tag] == label]
            plt.plot([x[label], x[label], x[label]], ac.iloc[:, 0], c=('firebrick' if rank[label] == 1 else 'royalblue'))
            plt.scatter(x[label], ac.iloc[0, 0], marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
        elif label == 'code_nosize':
            # print(df[df[tag] == label]) # mean mean-s mean+s
            cn = df[df[tag] == label]
            plt.plot([x[label], x[label], x[label]], cn.iloc[:, 0], c=('firebrick' if rank[label] == 1 else 'royalblue'))
            plt.scatter(x[label], cn.iloc[0, 0], marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
        elif label == 'cs_':
            # print(df[df[tag] == label]) # mean mean-s mean+s
            cs = df[df[tag] == label]
            plt.plot([x[label], x[label], x[label]], cs.iloc[:, 0], c=('firebrick' if rank[label] == 1 else 'royalblue'))
            plt.scatter(x[label], cs.iloc[0, 0], marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
    plt.xticks(list(x.values()), list(x.keys()))  # rotation为标签旋转角度
    plt.title(measure+'(SK_ESD_res)')
    plt.savefig("../../res/rq1_2-res/"+validation+"/sk_esd/"+tag+"_"+model+"_"+measure+".png")
    # plt.show()
    plt.close()


validation = "cross_validation"
# validation = "cross_project_validation"
sk = importr('ScottKnottESD')
tag = 't15'  # 只考虑t15
models = ['rf', 'lr', 'nb', 'xgb', 'svm']
for model in models:
    df = pd.read_csv("../../res/rq1_2-res/"+validation+"/pred_res/"+model+"/"+tag+".csv")
    data = sk_esdInput(df,model,tag,validation)

    data.dropna(axis=0,thresh=2,inplace=False)
    df1 = data[data[tag] == 'roc_auc']
    df2 = data[data[tag] == 'acc']
    df3 = data[data[tag] == 'mcc']

    res_img(df1,'roc_auc',tag,model,validation)
    res_img(df2,'acc',tag,model,validation)
    res_img(df3,'mcc',tag,model,validation)
