# 特征重要性平均值-汇总-数据
import pandas as pd
import matplotlib.pyplot as plt

def ShapData(path,label):
    df = pd.read_csv(path)
    Data = df[df['method'] == label].iloc[:, 1:]
    num = Data.isna().sum()
    for item in num.index.values:
        if num[item] > 8 * len(Data) / 10:  # 8/10的为null值，删除该列
            Data.drop(columns=item, inplace=True)
    fsiMean = Data.mean().sort_values(ascending=False).to_frame(name=label+'Mean')
    return fsiMean

if __name__ == '__main__':
    validation = ["within_project", "cross_project"]
    labels = ["all_code","cs_"]
    for v in validation:
        for label in labels:
            print(v,'----',label)
            path = "../../res/rq3-res/" + v + "/fsiSum/shap_fsiSum.csv"
            res = ShapData(path,label)
            res.to_csv("../../res/rq3-res/" + v + "/fsiSum/"+label+'Mean.csv')
            # break
        # break
