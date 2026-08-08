"""
Quant Memory Engine (QME) algorithm package.

Module layout:
    quant_memory_engine  - V3.1 core engine (state vector + retrieval + replay + revise + retain)
    state_vector         - 11-dim multi-subspace state-vector construction
    retrieval            - Mahalanobis-distance retrieval + decoupled temporal decay
    replay               - strategy replay engine
    revise               - parameter revision (strike / DTE / position adaptation)
    retain               - case-library update

    baselines/           - baseline methods
        dtw              - DTW dynamic time warping
        ms_garch         - MS-GARCH regime switching
        time2vec_knn     - Time2Vec + k-NN deep embedding
        cosine_kline     - cosine-similarity K-line

    metrics/             - evaluation metrics
        sharpe           - annualized Sharpe ratio
        dm_test          - Diebold-Mariano test
        spa_test         - Hansen SPA test
        win_rate         - win rate and profit/loss ratio
"""
