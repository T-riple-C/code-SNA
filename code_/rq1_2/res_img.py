# 绘图
import os
import random
from functools import reduce

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 多组箱型图
def multiBoxPlot(df,model,validation):
    data1 = []
    data2 = []
    data3 = []
    acc = df[['acc','t15']]
    roc_auc = df[['roc_auc', 't15']]
    mcc = df[['mcc', 't15']]
    labels = ["all_code", "code_nosize", "cs_"]
    for label in labels:
        # print(acc[acc['t15'] == label]['acc'].values)
        data1.append(acc[acc['t15'] == label]['acc'].values)
        data2.append(roc_auc[roc_auc['t15'] == label]['roc_auc'].values)  # roc_auc包含nan，则无法画箱型图
        data3.append(mcc[mcc['t15'] == label]['mcc'].values)

    # 三个箱型图的颜色 RGB （均为0~1的数据）
    # 黄花色：218，165，105；金黄色：255，215，0；淡黄色：245，222，179
    colors = [(218 / 255., 165 / 255., 105 / 255.), (255 / 255., 215 / 255., 0 / 255.),
             (245 / 255., 222 / 255., 179 / 255.)]
    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
    bplot1 = plt.boxplot(data1, patch_artist=True, labels=labels, positions=(1, 1.4, 1.8), widths=0.3)
    # 将三个箱分别上色
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    bplot2 = plt.boxplot(data2, patch_artist=True, labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3)

    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    bplot3 = plt.boxplot(data3, patch_artist=True, labels=labels, positions=(4, 4.4, 4.8), widths=0.3)

    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    plt.title(validation+'_'+model)

    x_position = [1, 2.5, 4]
    x_position_fmt = ["acc", "roc_auc", "mcc"]
    plt.xticks([i + 0.8 / 2 for i in x_position], x_position_fmt)

    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    plt.legend(bplot3['boxes'], labels, loc='lower left')  # 绘制表示框，右下角绘制
    plt.savefig("../../res/rq1_2-res/"+validation+"/pred_res/"+model+"/res.png")
    plt.show()

# todo roc_auc,mcc,acc 分开统计
def table(tag,models,cols,validation):
    roc_auc = pd.DataFrame(index=cols)
    acc = pd.DataFrame(index=cols)
    mcc = pd.DataFrame(index=cols)
    for model in models:
        data = pd.read_csv("../../res/rq1_2-res/"+validation+"/pred_res/" + model + '/' + tag + ".csv")
        tmp = data.groupby('t15', axis=0).mean()
        roc_auc[model] = tmp['roc_auc']
        acc[model] = tmp['acc']
        mcc[model] = tmp['mcc']
        # break
    roc_auc.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/roc_auc.csv")
    acc.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/acc.csv")
    mcc.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/mcc.csv")


if __name__ == '__main__':
    validation = "cross_validation"
    # validation = "cross_project_validation"
    tag = 't15'
    models = ['rf', 'lr', 'nb', 'xgb', 'svm']
    # 所有-表结果
    '''
    cols = ['all_code', 'code_nosize', 'cs_']
    table(tag,models,cols,validation)
    '''
    # 模型预测结果的箱型图
    for model in models:
        df = pd.read_csv("../../res/rq1_2-res/" + validation + "/pred_res/" + model + "/" + tag + ".csv")
        df.fillna(0, inplace=True)
        multiBoxPlot(df, model, validation)
        # break