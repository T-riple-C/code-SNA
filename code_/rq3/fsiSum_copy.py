import pandas as pd
import matplotlib.pyplot as plt

def ShapVariation(path,validation):
    df = pd.read_csv(path)
    df1 = df.iloc[:, 63:]   # 与原数据保持一致
    df1['pro'] = df['pro']
    df1['method'] = df['method']
    df1.dropna(axis=1, how='all', inplace=True)  # 删除值全为NAN的列
    df1.fillna(0,inplace=True)
    # print(df1) # print(df1.columns.values)  #包含原始col中所有的ck度量
    allCodeData = df1[df1['method'] == 'all_code'].iloc[:,0:-2].mean()
    nosizeData = df1[df1['method'] == 'code_nosize'].iloc[:,0:-2].mean()
    cs_Data = df1[df1['method'] == 'cs_'].iloc[:,0:-2].mean()

    x_name = allCodeData.index.values  # index一致
    x = range(len(x_name))
    plt.figure(figsize=(8, 5))
    plt.plot(x_name, allCodeData, '-', label='allcode')
    plt.plot(x_name, nosizeData, '--', label='nosize')
    plt.plot(x_name, cs_Data, '-.', label='cs')
    plt.legend()  # 让图例生效
    plt.xticks(x, x_name, rotation=30)

    plt.xlabel('metrics')
    plt.ylabel("feature importance")
    plt.title(validation+"-variation")
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/shap/variation_" + tag + ".png")
    plt.show()

def ShapData(path, tag, validation):
    df = pd.read_csv(path)
    # cs_数据的特征重要性——前10
    cs_Data = df[df['method'] == 'cs_'].iloc[:, 1:]
    cs_num = cs_Data.isna().sum()
    for item in cs_num.index.values:
        if cs_num[item] > 8 * len(cs_Data) / 10:
            cs_Data.drop(columns=item, inplace=True)
    res = cs_Data.mean().sort_values(ascending=False)
    res.to_csv("../../res/rq3-res/" + validation + "/fsiSum/shap/all_cs.csv")
    print(res)


# shap/sk-esd fsiSum，仅考虑rf/t15
if __name__ == '__main__':
    # validation = "cross_validation"
    validation = "cross_project_validation"
    tag = 't15'
    model = 'rf'
    path1 = "../../res/rq3-res/" + validation + "/fsiSum/" + tag + "-shap_fsiSum.csv"
    # path2 = "../../res/rq3-res/" + validation + "/fsiSum/" + tag + "-sk-esd_fsiSum.csv"

    # SHAP-特征重要性平均值——图
    # ShapData(path1, tag, validation)

    # SHAP-其余code度量特征重要性变化（三组：allcode/codenosize  codenosize/cs_  allcode/cs_）
    ShapVariation(path1,validation)
