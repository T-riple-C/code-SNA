相对功能文件——不用加入验证步骤
- fsi_importance_summary.py
  - 此文件里shap汇总/sk-esd汇总数据（shap汇总数据经sk-esd_summary计算得到）来自于shap_data
  - 关于总体特征重要性-allcode和cs_的绘图代码-可以实现——特征重要性和排名以数据形式呈现——见desktop/tmp.doc的示例表格数据
  - 关于两组特征变化情况——可以用折线图体现  
- sk-esd_rank1data.py
  - 统计了sk-esd的特征排名
  - 统计了sk-esd中特征出现排名1的特征的次数/比例——突出特征重要性  
- sk-esd_summary.py
    - 这个文件只是将shap的汇总结果用sk-esd的方法计算，得到新的sk-esd的汇总结果
    
