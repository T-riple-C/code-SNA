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
    df_new = pd.DataFrame(columns=cols)
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
    # plt.savefig("../../res/rq1_2-res/"+validation+"/sk_esd/"+tag+"_"+model+"_"+measure+".png")
    # plt.show()
    plt.close()


# todo:把三种评估指标—sk-esd关于模型性能和均值-绘制在同一张图上
#  按照评估指标划分
def res_img_1012(df,tag,model,validation):
    # print(df)
    df1 = df[df[tag] == 'roc_auc']
    df2 = df[df[tag] == 'acc']
    df3 = df[df[tag] == 'mcc']
    robjects.globalenv['df1'] = df1.iloc[:,0:-1]
    r_sk11 = sk.sk_esd(df1.iloc[:, 0:-1])  # roc_auc-模型分组信息
    # print(r_sk11[4])  # roc_auc (mean mean-s mean+s)
    robjects.globalenv['df2'] = df2.iloc[:, 0:-1]
    r_sk21 = sk.sk_esd(df2.iloc[:, 0:-1])  # acc-模型分组信息
    # print(r_sk21[4])
    robjects.globalenv['df3'] = df3.iloc[:, 0:-1]
    r_sk31 = sk.sk_esd(df3.iloc[:, 0:-1])  # mcc-模型分组信息
    # print(r_sk31[4])

    method = list(r_sk11[4].names[0])  # ['all_code','code_nosize', 'cs_']
    # print(method)
    m_value1 = list(r_sk11[4])
    m_value2 = list(r_sk21[4])
    m_value3 = list(r_sk31[4])
    data = pd.DataFrame(columns=['roc_auc','acc','mcc', tag])
    # 格式： 列名-评价指标+tag标签；值-以三种度量集分割；每种-三种度量集对应的mean mean-s mean+s
    # sk-esd模型性能值和分组信息分开，在绘制的时候才用作判断条件
    for i in range(0, len(method)):  # i method的index
        for j in range(0, 3):  # i+j*4_m_value的索引
            tmp = [[m_value1[i + j * 3], m_value2[i + j * 3], m_value3[i + j * 3], method[i]]]
            # method[i]对应三种m_value的 mean, mean-s, mean+s
            data = data.append(pd.DataFrame(tmp, columns=['roc_auc','acc','mcc', tag]), ignore_index=True)
        # break
    # print(data)
    # print(r_sk11[1].names,list(r_sk11[1]))

    # 位置按照评估指标划分;每种度量集+0.2
    x = {'roc_auc': 1, 'acc': 2, 'mcc': 3}
    # 按评估指标绘图——rank
    # '''
    measures = ['roc_auc','acc','mcc']
    for m in measures:
        rank = {}
        df = data[[m,tag]]
        print(df)
        if m == 'roc_auc':
            for k in range(0, len(list(r_sk11[1].names))):
                rank[list(r_sk11[1].names)[k]] = list(r_sk11[1])[k]
        elif m == 'acc':
            for k in range(0, len(list(r_sk21[1].names))):
                rank[list(r_sk21[1].names)[k]] = list(r_sk21[1])[k]
        elif m == 'mcc':
            for k in range(0, len(list(r_sk31[1].names))):
                rank[list(r_sk31[1].names)[k]] = list(r_sk31[1])[k]
        # print(list(rank.keys()))
        # print(rank)
        # '''
        for label in list(rank.keys()):
            if label == 'all_code':
                # print(df[df[tag] == label]) # mean mean-s mean+s
                ac = df[df[tag] == label]
                plt.plot([x[m]-0.2, x[m]-0.2, x[m]-0.2], ac.iloc[:, 0],marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
                # plt.scatter(x[m]-0.2, ac.iloc[0, 0], marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
            elif label == 'code_nosize':
                # print(df[df[tag] == label]) # mean mean-s mean+s
                cn = df[df[tag] == label]
                plt.plot([x[m], x[m], x[m]], cn.iloc[:, 0],marker='*', c=('firebrick' if rank[label] == 1 else 'royalblue'))
                # plt.scatter(x[m], cn.iloc[0, 0], marker='*', c=('firebrick' if rank[label] == 1 else 'royalblue'))
            elif label == 'cs_':
                # print(df[df[tag] == label]) # mean mean-s mean+s
                cs = df[df[tag] == label]
                plt.plot([x[m]+0.2, x[m]+0.2, x[m]+0.2], cs.iloc[:, 0],marker='^', c=('firebrick' if rank[label] == 1 else 'royalblue'))
                #plt.scatter(x[m]+0.2, cs.iloc[0, 0], marker='^', c=('firebrick' if rank[label] == 1 else 'royalblue'))
        #'''
    # plt.show()
    # '''
    plt.xticks(list(x.values()), list(x.keys()))  # rotation为标签旋转角度
    plt.title("SK-ESD-"+validation)
    plt.legend(method, loc='lower left')  # 绘制表示框，左下角绘制
    plt.savefig("../../res/rq1_2-res/"+validation+"/sk_esd/"+tag+"_"+model+"_skESD-Res.png")
    plt.show()
    # plt.close()
    # '''

validation = "cross_validation"
# validation = "cross_project_validation"
sk = importr('ScottKnottESD')
tag = 't15'  # 只考虑t15
models = ['rf', 'lr', 'nb', 'xgb', 'svm']
for model in models:
    df = pd.read_csv("../../res/rq1_2-res/"+validation+"/pred_res/"+model+"/"+tag+".csv")
    data = sk_esdInput(df,model,tag,validation)

    data.dropna(axis=0,thresh=2,inplace=False)

    res_img_1012(data, tag, model, validation)
    break  # 仅rf的结果