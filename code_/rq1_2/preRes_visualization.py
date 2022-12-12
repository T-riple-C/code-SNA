# 模型预测结果可视化-图/表
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt


# 多组箱型图
def multiBoxPlot(df,model,validation,tag,labels):
    data1 = []
    data2 = []
    data3 = []
    acc = df[['acc',tag]]
    roc_auc = df[['roc_auc', tag]]
    mcc = df[['mcc', tag]]
    for label in labels:
        data1.append(acc[acc[tag] == label]['acc'].values)
        data2.append(roc_auc[roc_auc[tag] == label]['roc_auc'].values)
        data3.append(mcc[mcc[tag] == label]['mcc'].values)

    # 三个箱型图的颜色 RGB （均为0~1的数据）
    # 黄花色：218，165，105；金黄色：255，215，0；淡黄色：245，222，179
    colors = [(218 / 255., 165 / 255., 105 / 255.), (255 / 255., 215 / 255., 0 / 255.),
             (245 / 255., 222 / 255., 179 / 255.)]
    # 绘制箱型图
    # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
    # 将三个箱分别上色
    bplot1 = plt.boxplot(data1, patch_artist=True, labels=labels, positions=(1, 1.4, 1.8), widths=0.3)
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


def table(tag,df,roc_auc,acc,mcc):
    tmp = df.groupby(tag, axis=0).mean()
    roc_auc[model] = tmp['roc_auc']
    acc[model] = tmp['acc']
    mcc[model] = tmp['mcc']


if __name__ == '__main__':
    # validation = "within-project"
    validation = "cross_project"
    tag = 'metrics'
    models = ['rf', 'lr', 'nb', 'xgb', 'svm']
    labels = ['all_code', 'code_nosize', 'cs_']

    ROC_AUC = pd.DataFrame(index=labels)
    ACC = pd.DataFrame(index=labels)
    MCC = pd.DataFrame(index=labels)

    for model in models:
        df = pd.read_csv("../../res/rq1_2-res/" + validation + "/pred_res/" + model + "/predRes.csv")

        # 汇总各模型 预测结果
        table(tag,df,ROC_AUC, ACC, MCC)

        # 模型预测结果的箱型图
        df.fillna(0, inplace=True)
        multiBoxPlot(df, model, validation,tag,labels)
        # break
    ROC_AUC.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/roc_auc.csv")
    ACC.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/acc.csv")
    MCC.to_csv("../../res/rq1_2-res/"+validation+"/pred_res/mcc.csv")
    # print(roc_auc,'\n',acc,'\n',mcc)