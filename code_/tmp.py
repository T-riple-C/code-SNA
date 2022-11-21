# tmp 临时文件用于实现、测试相关代码块
from itertools import permutations
from itertools import chain, combinations
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns
import pandas as pd
import os
from code_.fun import config
import warnings
warnings.filterwarnings("ignore")


# 列出文件目录
'''
validation = "cross_validation"
for path, catalogue, name in os.walk("../res/rq3-res/" + validation + "/sk_esd_rank/"):
    print(name)
'''


# 排列组合
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# concat 拼接csv文件
def concat_csv(files):
    df_list = []
    for file in files:
        print(file)
        fp = '../res/test/t15/' + file
        df = pd.read_csv(fp, low_memory=False)  # 自动读取n个csv格式文件
        df_list.append(df)  # 每读取一个csv格式文件，都存到列表当中。最终我们得到['table1', 'table2', ... , 'tableN']
        df_final = reduce(lambda left, right: pd.merge(left, right, on='fname', how='outer'), df_list)
        df_final.to_csv('../res/test/t15/rf_concat.csv', index=False)


# 判断列是否全为NAN值
def isNull(df):
    for cols in df.columns.values:
        if df[cols].isnull().all():
            df.drop(columns=cols, inplace=True)
    print(df)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

dataset = config.datasets
tag = 't15'  # 只考虑t15
# cols = ['all_code', 'code_nosize', 'cs_']
# col = 'cs_'
col = 'codeSNA'
validation_cp = "cross_project_validation"
validation_wp = "cross_validation"

# 设变量名：wp_shap;cp_shap;wp_rank;cp_rank
wp_shap_path = "../res/rq3-res/" + validation_wp + "/fsiSum/t15-shap_fsiSum.csv"  # shap值
wp_shap = pd.read_csv(wp_shap_path)
wp_shap= wp_shap[wp_shap['method'] == col]
cp_shap_path = "../res/rq3-res/" + validation_cp + "/fsiSum/t15-shap_fsiSum.csv"  # shap值
cp_shap = pd.read_csv(cp_shap_path)
cp_shap= cp_shap[cp_shap['method'] == col]

wp_rank_path = "../res/rq3-res/" + validation_wp + "/sk_esd_rank/"+col+"_t15-rank.csv"   # cs_rank
wp_rank = pd.read_csv(wp_rank_path)
cp_rank_path = "../res/rq3-res/" + validation_cp + "/sk_esd_rank/"+col+"_t15-rank.csv"   # cs_rank
cp_rank = pd.read_csv(cp_rank_path)


for i in range(0,len(dataset)):
    wp_shapley = wp_shap[wp_shap['pro'] == dataset[i][0:-11]]   # pro 比rank多了一个method
    cp_shapley = cp_shap[cp_shap['pro'] == dataset[i][0:-11]]
    wp_r = wp_rank[wp_rank['pro'] == dataset[i][0:-11]]   # 这个数据的源文件是没有pro-**列名**-的
    cp_r = cp_rank[cp_rank['pro'] == dataset[i][0:-11]]

    wp_shapley.drop(columns=['pro','method'],inplace=True)
    cp_shapley.drop(columns=['pro', 'method'], inplace=True)
    wp_r.drop(columns=['pro'],inplace=True)
    wp_r.columns=config.cols
    cp_r.drop(columns=['pro'], inplace=True)
    cp_r.columns = config.cols

    # res1:以wp_shap_为mean栏-并排序，补充两种验证方式下的sk-esd排名数据(data:wp_shapley,wp_r,cp_r)
    # todo
    '''
    res1 = wp_shapley.append(wp_r,ignore_index=True)
    res1 = res1.append(cp_r,ignore_index=True)
    res1 = res1.sort_values(axis=1,by=0,ascending=False)
    res1.index = ['wp_shap','wp_r','cp_r']
    # print(res1.T)
    res1.T.to_csv("../res/rq3-res/wp_shap/"+dataset[i][0:-11]+".csv")
    '''
    # res2: 以两种验证方式下mean_shap为mean栏-并排序，补充两种验证方式下的sk-esd排名数据(data:mean(wp_shapley+cp_shapley,wp_r,cp_r))
    res2 = wp_shapley.append(cp_shapley,ignore_index=True)
    # print(res2.mean(axis=0).to_frame().T)
    tmp = res2.mean(axis=0).to_frame().T
    res2 = tmp.append(wp_r,ignore_index=True)
    res2 = res2.append(cp_r,ignore_index=True)
    res2 = res2.sort_values(axis=1, by=0, ascending=False)
    res2.index = ['mean_shap', 'wp_r', 'cp_r']
    print(res2.T)
    res2.T.to_csv("../res/rq3-res/mean_shap/" + dataset[i][0:-11] + ".csv")
    # break

