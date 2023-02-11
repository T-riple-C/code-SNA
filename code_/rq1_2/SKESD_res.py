import numpy as np
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
pandas2ri.activate()
import pandas as pd
from matplotlib import pyplot as plt

# 模型预测结果数据结构——SKESD计算
def sk_esdInput(df,tag):
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
    return df_new


# 可视化SKESD-结果（多个评估指标）
def multi_res_img(df,tag,model,validation):
    df1 = df[df[tag] == 'roc_auc']
    df2 = df[df[tag] == 'acc']
    df3 = df[df[tag] == 'mcc']
    robjects.globalenv['df1'] = df1.iloc[:,0:-1]
    r_sk11 = sk.sk_esd(df1.iloc[:, 0:-1])  # roc_auc-模型分组信息
    robjects.globalenv['df2'] = df2.iloc[:, 0:-1]
    r_sk21 = sk.sk_esd(df2.iloc[:, 0:-1])  # acc-模型分组信息
    robjects.globalenv['df3'] = df3.iloc[:, 0:-1]
    r_sk31 = sk.sk_esd(df3.iloc[:, 0:-1])  # mcc-模型分组信息

    method = list(r_sk11[4].names[0])  # ['all_code','code_nosize', 'cs_']
    m_value1 = list(r_sk11[4])
    m_value2 = list(r_sk21[4])
    m_value3 = list(r_sk31[4])
    data = pd.DataFrame(columns=['roc_auc','acc','mcc', tag])
    # data格式： 列名-评价指标+tag标签；值-以三种度量集分割；每-对应的mean mean-s mean+s
    for i in range(0, len(method)):
        for j in range(0, 3):
            tmp = [[m_value1[i + j * 3], m_value2[i + j * 3], m_value3[i + j * 3], method[i]]]
            data = data.append(pd.DataFrame(tmp, columns=['roc_auc','acc','mcc', tag]), ignore_index=True)
    # 位置：按照评估指标划分;每种度量集+0.2
    x = {'roc_auc': 1, 'acc': 2, 'mcc': 3}
    # x = {'AUC-ROC': 1, 'ACC': 2, 'MCC': 3}
    # 按评估指标绘图——rank
    measures = ['roc_auc','acc','mcc']
    # measures = ['AUC-ROC', 'ACC', 'MCC']
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
        # '''
        for label in list(rank.keys()):
            if label == 'all_code':
                ac = df[df[tag] == label]
                plt.plot([x[m]-0.2, x[m]-0.2, x[m]-0.2], ac.iloc[:, 0],marker='o', c=('firebrick' if rank[label] == 1 else 'royalblue'))
            elif label == 'code_nosize':
                cn = df[df[tag] == label]
                plt.plot([x[m], x[m], x[m]], cn.iloc[:, 0],marker='*', c=('firebrick' if rank[label] == 1 else 'royalblue'))
            elif label == 'cs_':
                cs = df[df[tag] == label]
                plt.plot([x[m]+0.2, x[m]+0.2, x[m]+0.2], cs.iloc[:, 0],marker='^', c=('firebrick' if rank[label] == 1 else 'royalblue'))
        # '''
    # '''
    for idx in range(0,len(method)):
        if method[idx] == 'cs_':
            method[idx] =  "-Size/+SNA" #"With SNA Metrics"
        elif method[idx] == 'all_code':
            method[idx] = "+Size/-SNA" # Without SNA Metrics
        elif method[idx] == 'code_nosize':
            method[idx] = "-Size/-SNA" #"With Code(Without Size) Metrics"
    # plt.xticks(list(x.values()), list(x.keys()))
    plt.xticks(list(x.values()),['AUC-ROC','ACC','MCC'])
    # plt.title("SK-ESD-"+validation)
    plt.legend(method, loc='lower left')  # 绘制表示框，左下角绘制
    plt.savefig("../../res/rq1_2-res/"+validation+"/sk_esd/"+model+"_SKESD-Res.png")
    # plt.show()
    plt.close() # 防止相互影响
    # '''


if __name__ == '__main__':
    validation = "within-project"
    # validation = "cross_project"
    sk = importr('ScottKnottESD')
    tag = 'metrics'
    models = ['rf', 'lr', 'nb', 'xgb', 'svm']
    for model in models:
        df = pd.read_csv("../../res/rq1_2-res/" + validation + "/pred_res/" + model + "/predRes.csv")

        data = sk_esdInput(df,tag)
        data.dropna(axis=0,thresh=2,inplace=False)

        multi_res_img(data, tag, model, validation)
        # # break
    print("done.")