# config：一些常变量

datasets = ['ant_1.4_concat.csv', 'ant_1.5_concat.csv', 'ant_1.6_concat.csv', 'ant_1.7_concat.csv',
            'ivy_2.0_concat.csv', 'jEdit_3.2.1_concat.csv', 'jEdit_4.0_concat.csv', 'jEdit_4.1_concat.csv',
            'jEdit_4.2_concat.csv', 'log4j_1.0.4_concat.csv',
            'lucene_2.0_concat.csv', 'lucene_2.2_concat.csv', 'lucene_2.4_concat.csv',
            'poi_1.5_concat.csv', 'poi_2.0_concat.csv', 'poi_2.5_concat.csv', 'poi_3.0_concat.csv',
            'synapse_1.0_concat.csv', 'synapse_1.1_concat.csv', 'synapse_1.2_concat.csv', 'Tomcat_6.0.39_concat.csv',
            'velocity_1.4_concat.csv', 'velocity_1.5_concat.csv', 'velocity_1.6.1_concat.csv', 'xalan_2.4.0_concat.csv',
            'xalan_2.5.0_concat.csv', 'xalan_2.6.0_concat.csv', 'xerces_1.2.0_concat.csv',
            'xerces_1.3.0_concat.csv', 'xerces_1.4.4_concat.csv']

# 原数据格式保持一致列
cols = ['Size(in)', 'Ties(in)', 'Pairs(in)', 'Densit(in)', 'AvgDis(in)', 'Diamet(in)', 'nWeakC(in)', 'pWeakC(in)',
        '2StepR(in)', '2StepP(in)', 'ReachE(in)', 'Broker(in)', 'nBroke(in)', 'EgoBet(in)', 'nEgoBe(in)', 'Size(out)',
        'Ties(out)', 'Pairs(out)', 'Densit(out)', 'AvgDis(out)', 'Diamet(out)', 'nWeakC(out)', 'pWeakC(out)',
        '2StepR(out)', '2StepP(out)', 'ReachE(out)', 'Broker(out)', 'nBroke(out)', 'EgoBet(out)', 'nEgoBe(out)',
        'Size(un)', 'Ties(un)', 'Pairs(un)', 'Densit(un)', 'AvgDis(un)', 'Diamet(un)', 'nWeakC(un)', 'pWeakC(un)',
        '2StepR(un)', '2StepP(un)', 'ReachE(un)', 'Broker(un)', 'nBroke(un)', 'EgoBet(un)', 'nEgoBe(un)', 'Degree_x',
        'EffSize', 'Efficienc', 'Constrain', 'Hierarchy', 'EgoBetwe', 'Ln(Constr', 'Indirects', 'Density', 'Degree_y',
        'Closeness', 'Betweenness', 'Eigenvector', 'EffSize(g)', 'Efficie(g)', 'Constra(g)', 'Hierarc(g)', 'Indirec(g)',
        'wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm',
        'amc', 'max(cc)', 'avg(cc)']

cols_skesd = ['Size.in.', 'Ties.in.', 'Pairs.in.', 'Densit.in.', 'AvgDis.in.', 'Diamet.in.', 'nWeakC.in.', 'pWeakC.in.',
              'X2StepR.in.', 'X2StepP.in.', 'ReachE.in.', 'Broker.in.', 'nBroke.in.', 'EgoBet.in.', 'nEgoBe.in.',
              'Size.out.', 'Ties.out.', 'Pairs.out.', 'Densit.out.',
              'AvgDis.out.', 'Diamet.out.', 'nWeakC.out.', 'pWeakC.out.', 'X2StepR.out.', 'X2StepP.out.', 'ReachE.out.',
              'Broker.out.', 'nBroke.out.', 'EgoBet.out.', 'nEgoBe.out.', 'Size.un.', 'Ties.un.', 'Pairs.un.',
              'Densit.un.', 'AvgDis.un.', 'Diamet.un.', 'nWeakC.un.', 'pWeakC.un.', 'X2StepR.un.', 'X2StepP.un.',
              'ReachE.un.', 'Broker.un.', 'nBroke.un.', 'EgoBet.un.', 'nEgoBe.un.', 'Degree_x', 'EffSize', 'Efficienc',
              'Constrain', 'Hierarchy', 'EgoBetwe', 'Ln.Constr', 'Indirects', 'Density', 'Degree_y', 'Closeness',
              'Betweenness', 'Eigenvector', 'EffSize.g.', 'Efficie.g.', 'Constra.g.', 'Hierarc.g.', 'Indirec.g.','wmc',
              'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic', 'cbm',
              'amc', 'max.cc.', 'avg.cc.']

code_cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam', 'ic',
             'cbm', 'amc', 'max(cc)', 'avg(cc)']