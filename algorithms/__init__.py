"""
PromptQuant 论文算法包

模块结构:
    quant_memory_engine  - V3.1 核心引擎（状态向量 + 检索 + 回放 + 修正 + 留存）
    state_vector         - 12维多子空间状态向量构造
    retrieval            - 马氏距离检索 + 解耦时间衰减
    replay               - 策略回放引擎（对接 option_backtester）
    revise               - 参数修正（行权价/DTE/仓位适配）
    retain               - 案例库更新

    baselines/           - 基线方法
        dtw              - DTW 动态时间规整
        ms_garch         - MS-GARCH 状态切换
        time2vec_knn     - Time2Vec + k-NN 深度嵌入
        cosine_kline     - 余弦相似度 K 线

    metrics/             - 评估指标
        sharpe           - 年化夏普比率
        dm_test          - Diebold-Mariano 检验
        spa_test         - Hansen SPA 检验
        win_rate         - 胜率与盈亏比
"""
