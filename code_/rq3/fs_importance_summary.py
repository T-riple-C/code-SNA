import pandas as pd
import matplotlib.pyplot as plt


def getShapData_1(path):
    # 比较——删除size度量前后的变化——code度量-0:19
    df = pd.read_csv(path)
    df1 = df.iloc[:, 63:]   # 与原数据保持一致
    df1['pro'] = df['pro']
    df1['method'] = df['method']
    df1 = df1.drop(axis=1, index=df1[df1['method'] == 'cs_'].index.values)
    df1.dropna(axis=1, how='all', inplace=True)  # 删除值全为NAN的列
    # print(df1)
    allCodeData = df1[df1['method'] == 'all_code'].reset_index()
    nosizeData = df1[df1['method'] == 'code_nosize'].reset_index()
    res = nosizeData.iloc[:, 0:-2] - allCodeData.iloc[:, 0:-2]  # 索引对齐的方式
    # print(res.mean()[1:])
    return res.mean()[1:].sort_values(ascending=False)


def getShapData_2(path):
    # 比较——加入SNA后与没有删除size度量的重要性变化
    df = pd.read_csv(path)
    df1 = df.iloc[:, 63:]
    df1['pro'] = df['pro']
    df1['method'] = df['method']
    df1 = df1.drop(axis=1, index=df1[df1['method'] == 'code_nosize'].index.values)
    df1.dropna(axis=1, how='all', inplace=True)  # 删除值全为NAN的列
    allCodeData = df1[df1['method'] == 'all_code'].reset_index()
    cs_Data = df1[df1['method'] == 'cs_'].reset_index()
    res = cs_Data.iloc[:, 0:-2] - allCodeData.iloc[:, 0:-2]
    # print(res.mean()[1:])
    return res.mean()[1:].sort_values(ascending=False)


def ShapData(path, tag, validation):
    df = pd.read_csv(path)
    '''
    # cs_数据的特征重要性——前10
    cs_Data = df[df['method'] == 'cs_'].iloc[:, 1:]
    cs_num = cs_Data.isna().sum()
    for item in cs_num.index.values:
        if cs_num[item] > 8 * len(cs_Data) / 10:
            cs_Data.drop(columns=item, inplace=True)
    print(cs_Data.mean().sort_values(ascending=False))
    plt.plot(cs_Data.mean().sort_values(ascending=False))
    plt.xticks(color='blue', rotation=-85)
    plt.title(tag)
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/shap/" + tag + '-all_cs.png')
    # plt.show()
    plt.close()

    # all_code数据的特征重要性——前10
    all_codeData = df[df['method'] == 'all_code'].iloc[:, 1:]
    all_num = all_codeData.isna().sum()
    for item in all_num.index.values:
        if all_num[item] > 8 * len(all_codeData) / 10:
            all_codeData.drop(columns=item, inplace=True)
    plt.plot(all_codeData.mean().sort_values(ascending=False))
    plt.xticks(color='blue', rotation=-85)
    plt.title(tag)
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/shap/" + tag + '-all_code.png')
    # plt.show()
    plt.close()
    '''

    # codeSNA-汇总数据的特征重要性-均值——前10
    codeSNA_Data = df[df['method'] == 'codeSNA'].iloc[:, 1:]
    codeSNA_num = codeSNA_Data.isna().sum()
    for item in codeSNA_num.index.values:
        if codeSNA_num[item] > 8 * len(codeSNA_Data) / 10:
            codeSNA_Data.drop(columns=item, inplace=True)
    print(codeSNA_Data.mean().sort_values(ascending=False))
def getSKData_1(path):
    # 比较——删除size度量前后的code度量特征重要性变化
    df = pd.read_csv(path)
    df1 = df.iloc[:, 63:]  # 与原数据保持一致
    # df1 = df.iloc[:, 42:]  # 在最后保存sk-esd结果时，删除了全为NULL的列，列数与SHAP——fsiSum不同，索引不一样
    df1['pro'] = df['pro']
    df1['method'] = df['method']
    # print(df1)
    df1 = df1.drop(axis=1, index=df1[df1['method'] == 'c_s_'].index.values)
    df1 = df1.drop(axis=1, index=df1[df1['method'] == 'cs_'].index.values)
    df1.dropna(axis=1, how='all', inplace=True)  # 删除值全为NAN的列
    # print(df1)
    allCodeData = df1[df1['method'] == 'all_code'].reset_index()
    nosizeData = df1[df1['method'] == 'code_nosize'].reset_index()
    res = nosizeData.iloc[:, 0:-2] - allCodeData.iloc[:, 0:-2]  # 索引对齐的方式
    # print(res.mean()[1:].sort_values(ascending=False))
    return res.mean()[1:].sort_values(ascending=False)


def getSKData_2(path):
    # 比较——加入SNA后与没有删除size度量的重要性变化
    df = pd.read_csv(path)
    df1 = df.iloc[:, 63:]  # 与原数据保持一致
    # df1 = df.iloc[:, 42:]
    df1['pro'] = df['pro']
    df1['method'] = df['method']
    df1 = df1.drop(axis=1, index=df1[df1['method'] == 'code_nosize'].index.values)
    df1.dropna(axis=1, how='all', inplace=True)  # 删除值全为NAN的列
    # print(df1)
    allCodeData = df1[df1['method'] == 'all_code'].reset_index()
    cs_Data = df1[df1['method'] == 'cs_'].reset_index()
    res = cs_Data.iloc[:, 0:-2] - allCodeData.iloc[:, 0:-2]
    # print(res.mean()[1:].sort_values(ascending=False))
    return res.mean()[1:].sort_values(ascending=False)


def SKData(path, tag, validation):
    df = pd.read_csv(path)
    # cs_数据的特征重要性——前10
    cs_Data = df[df['method'] == 'cs_'].iloc[:, 1:]
    cs_num = cs_Data.isna().sum()
    # print(cs_num)
    for item in cs_num.index.values:
        if cs_num[item] > 8 * len(cs_Data) / 10:
            cs_Data.drop(columns=item, inplace=True)
    plt.plot(cs_Data.mean().sort_values(ascending=False))
    plt.xticks(color='blue', rotation=-85)
    plt.title(tag)
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/sk-esd/" + tag + '-all_cs.png')
    # plt.show()
    plt.close()

    all_codeData = df[df['method'] == 'all_code'].iloc[:, 1:]
    all_num = all_codeData.isna().sum()
    for item in all_num.index.values:
        if all_num[item] > 8 * len(cs_Data) / 10:
            all_codeData.drop(columns=item, inplace=True)
    plt.plot(all_codeData.mean().sort_values(ascending=False))
    plt.xticks(color='blue', rotation=-85)
    plt.title(tag)
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/sk-esd/" + tag + '-all_code.png')
    # plt.show()
    plt.close()


def draw(data, tag, validation, method, label='size'):
    plt.plot(data)
    plt.xticks(color='blue', rotation=60)
    plt.title(tag)
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/" + method + '/variation_' + tag + '_' + label + '.png')
    # plt.show()
    plt.close()

def draw_subfig(data1,data2, tag, validation, method):
    plt.figure(figsize=(10, 5))
    plt.suptitle(validation)
    plt.subplot(1, 2, 1)
    plt.plot(data1)
    plt.xticks(color='blue', rotation=45)
    plt.title("all_code-codeNosize")

    plt.subplot(1,2,2)
    plt.plot(data2)
    plt.xticks(color='blue', rotation=45)
    plt.title("all_code-cs")
    # 显示画布
    plt.savefig("../../res/rq3-res/" + validation + "/fsiSum/" + method + '/variation.png')
    plt.show()


# shap/sk-esd fsiSum，仅考虑rf/t15
if __name__ == '__main__':
    validation = "cross_validation"
    # validation = "cross_project_validation"
    tag = 't15'
    model = 'rf'
    path1 = "../../res/rq3-res/" + validation + "/fsiSum/" + tag + "-shap_fsiSum.csv"
    path2 = "../../res/rq3-res/" + validation + "/fsiSum/" + tag + "-sk-esd_fsiSum.csv"
    # 仅为汇总-特征重要性平均值-数据
    ShapData(path1, tag, validation)

    '''
    # SHAP-特征重要性平均值——图
    # ShapData(path1, tag, validation)    # 两张图
    # SHAP的结果_特征重要性变化
    data = getShapData_1(path1)
    cs_data = getShapData_2(path1)
    # draw(data, tag, validation, 'shap', 'size')     # 一张图
    # draw(cs_data, tag, validation, 'shap', 'cs_')   # 一张图
    draw_subfig(data,cs_data,tag,validation,'shap')
    '''
    '''
    # SK-ESD特征重要性平均值——图
    SKData(path2,tag,validation)
    # SK-ESD的结果_特征重要性变化
    data = getSKData_1(path2)
    cs_data = getSKData_2(path2)
    draw(data, tag, validation, 'sk-esd', 'size')
    draw(cs_data, tag, validation, 'sk-esd', 'cs_')
    '''
