### 一、 确认：币安 (Binance) 的数据确实全免费！

**是的，你的理解完全正确！** 币安的市场公开数据（K线、永续合约资金费率、持仓量、大户多空比等）**100% 免费**，并且有两种极简获取方式：

1. **批量文件直接下载（最省事）：**
币安官方提供了 [data.binance.vision](https://data.binance.vision/) 网站，你可以直接按年份/月份下载 Zip 压缩包（包含 CSV 格式的历史 K 线、资金费率等），不需要写代码。
2. **API 接口实时抓取：**
通过 Python 调用 API (`GET /api/v3/klines` 或 `GET /fapi/v1/fundingRate`)，**完全不需要注册账号，也不需要配置 API Key**，直接发送 Request 请求就能拿到 JSON 数据。

---

### 二、 关键问题：Deribit 的数据怎么搞？

好消息是：**Deribit 的公开行情数据（DVOL 波动率指数、期权链 Mark IV、希腊字母 Greeks）同样是 100% 免费的，且不需要任何 API Key 或登录！**

你只需要像调普通网页一样，调用它的公开 API（Public Endpoints）就能把 5 年的历史数据拉下来。

下面为您梳理 **一步一步（Step-by-Step）的操作指引**：

---

### 🛠️ Deribit 数据获取 4 步走操作指南

#### 第一步：准备运行环境（只需 3 分钟）

在你或团队成员的电脑（或服务器）上安装好 Python 3.9+，并安装两个数据处理的基础包：

```bash
pip install requests pandas pyarrow

```

---

#### 第二步：抓取 DVOL 历史波动率指数（衍生品子空间 $\mathcal{S}_{vol}$ 核心）

Deribit 官方提供了专门的 DVOL 指数 API 端点。运行以下 Python 脚本，即可一次性把过去 5 年的 BTC/ETH DVOL 历史日线全部下载并保存为 CSV：

```python
import requests
import pandas as pd
import datetime

def fetch_deribit_dvol(currency="BTC", start_year=2021, end_year=2025):
    """
    抓取 Deribit 历史 DVOL (波动率指数) 日线数据
    """
    start_ts = int(datetime.datetime(start_year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.datetime(end_year, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
        "resolution": "1D" # 日线粒度
    }
    
    print(f"正在抓取 {currency} DVOL 数据 ({start_year}-{end_year})...")
    res = requests.get(url, params=params).json()
    
    if "result" in res and "data" in res["result"]:
        # 返回字段: [timestamp, open, high, low, close]
        raw_data = res["result"]["data"]
        df = pd.DataFrame(raw_data, columns=["timestamp", "open", "high", "low", "dvol_close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.date
        df["currency"] = currency
        
        # 只保留所需字段
        df = df[["date", "currency", "dvol_close"]]
        print(f"✅ 成功抓取 {len(df)} 条 {currency} DVOL 数据！")
        return df
    else:
        print(f"❌ 抓取失败: {res}")
        return pd.DataFrame()

# 运行抓取
btc_dvol = fetch_deribit_dvol("BTC", 2021, 2025)
eth_dvol = fetch_deribit_dvol("ETH", 2021, 2025)

# 合并并保存为本地文件
dvol_df = pd.concat([btc_dvol, eth_dvol])
dvol_df.to_csv("deribit_dvol_2021_2025.csv", index=False)
print("💾 数据已保存至: deribit_dvol_2021_2025.csv")

```

---

#### 第三步：获取期权链快照与 Greeks（回测引擎与历史特征库）

对于每日期权链快照（109 万条数据），Deribit 提供了 `public/get_book_summary_by_currency` 端点。

有两种落地方案：

##### 方案 A（全自动免费 API 遍历脚本）：

通过 Python 循环抓取每日快照，将每日 UTC 23:59:59 的在售期权抓取下来：

```python
import requests
import pandas as pd

def fetch_current_option_chain(currency="BTC"):
    """
    获取 Deribit 当前所有挂牌期权合约的 Mark Price, Mark IV 和 Greeks 快照
    """
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {
        "currency": currency,
        "kind": "option"
    }
    res = requests.get(url, params=params).json()
    
    records = []
    if "result" in res:
        for item in res["result"]:
            # 提取关键列
            records.append({
                "instrument_name": item.get("instrument_name"),
                "underlying_price": item.get("underlying_price"),
                "mark_price": item.get("mark_price"),
                "mark_iv": item.get("mark_iv"),
                "bid_price": item.get("bid_price"),
                "ask_price": item.get("ask_price"),
                "volume": item.get("volume"),
                "open_interest": item.get("open_interest"),
                "creation_timestamp": item.get("creation_timestamp")
            })
    return pd.DataFrame(records)

# 抓取当前快照示例
df_chain = fetch_current_option_chain("BTC")
print(df_chain.head())

```

##### 方案 B（快捷历史数据包下载）：

如果你嫌按天遍历 API 速度慢，行业内通常会直接去下载第三方开源的 **Deribit 历史归档数据集**（比如 [Tardis.dev Free Datasets](https://tardis.dev/) 或 [Amberdata Archive](https://www.amberdata.io/) 的免费样例），或者在 GitHub 搜索 `deribit-historical-data`，里面会有量化爱好者整理好的 2021-2025 年 Deribit 日终 CSV 数据包，几分钟就能直接下载解压完毕。

---

#### 第四步：数据清洗与对齐 CheckList

拿到 CSV/Parquet 文件后，只需让编程助手在数据入库前检查三件事：

1. **时间截帧对齐**：确保 Deribit 的数据与 Binance 的 K 线数据全部按 **`UTC 23:59:59`** 强制切片对齐。
2. **剔除僵尸合约**：过滤掉 `open_interest == 0` 或 `volume == 0` 的零流动性极端虚值期权，避免影响回测准确度。
3. **压缩保存为 Parquet 格式**：使用 `df.to_parquet("deribit_options.parquet")` 保存。Parquet 格式读取速度比 CSV 快 20 倍，且压缩率高达 80%（1.2GB 数据压缩后仅需 ~180MB）。

---

### 💡 总结

1. **币安（Binance）数据**：去 [data.binance.vision](https://data.binance.vision/) 直接下 CSV，或者调 API，完全免费。
2. **Deribit 数据**：直接运行上面的脚本，免登录、免 API Key，直接抓取 DVOL 与期权快照！

您可以把第二步的代码直接发给编程助手，让他执行脚本把数据保存在 `data/raw/` 目录下即可！


📊 PromptQuant 12 维市场状态特征与期权回测全量数据获取规范

本文档针对论文 paper_draft_2.pdf 中定义的 12 维多子空间市场状态向量（$\mathcal{S}_{vol}, \mathcal{S}_{mkt}, \mathcal{S}_{mic}, \mathcal{S}_{flow}$） 以及 期权策略回放引擎（Replay Engine） 的数据需求，提供精准的数据采集清单、API 接口映射、记录条数计算及预处理要求。

📌 一、 全局参数与数据规模汇总

数据时间跨度：2021-01-01 00:00:00 UTC 至 2025-12-31 23:59:59 UTC（共 5 年 / 1,826 天）。

标的资产（Underlying Assets）：BTC（比特币）与 ETH（以太坊）。

采样频率：日频（Daily UTC 24:00 截帧）。

预估总存储空间：Parquet / Compressed CSV 压缩后约 160 MB ~ 210 MB（解压原始 JSON 约 1.2 GB ~ 1.5 GB）。

📐 二、 12 维特征向量数据采集全表

以下 12 个维度直接构成案例检索中的输入向量 $V_t \in \mathbb{R}^{12}$。在 2021-2025 期间，每个特征维度对应 3,652 条日频数据（1,826 天 × 2 币种）。

子空间 (Subspace)

特征代码

特征名称与业务含义

数据源与建议 API / 端点

API 关键请求参数与返回字段

样本条数

计算公式 / 预处理要求

衍生品子空间 $\mathcal{S}_{vol}$

IVP

隐含波动率百分位



(DVOL Percentile)

Deribit API



public/get_volatility_index_data

currency: BTC/ETH



resolution: 1D



返回: dvol 指数

3,652 条

$IVP_t = \text{PercentileRank}_{365d}(\text{DVOL}_t)$



(基于过去 365 天滚动排名归一化至 $[0, 1]$)



VRP

波动率风险溢价



(Volatility Risk Premium)

Deribit API + Binance API

dvol (Deribit)



close 价格 (Binance 现货)

3,652 条

$VRP_t = \text{DVOL}_t - HV_t^{(20d)}$



其中 $HV^{(20d)}$ 为 20 日年化历史波动率



Slope

期限结构斜率



(Term Structure Slope)

Deribit API



public/get_historical_volatility 或期权链快照

提取 DTE=30 天与 DTE=7 天 ATM 期权的 Mark IV

3,652 条

$Slope_t = IV_t^{(30d)} - IV_t^{(7d)}$



(正值代表远月升水，负值代表近月倒挂)



Skew

25-Delta 风险逆转



(25-Delta Skew)

Deribit API



public/get_book_summary_by_currency

筛选相同到期日下 $\Delta = +0.25$ Call 与 $\Delta = -0.25$ Put

3,652 条

$Skew_t = IV_t(\Delta_{25C}) - IV_t(\Delta_{25P})$



(衡量市场尾部看涨 vs 看跌情绪偏斜)

动量子空间 $\mathcal{S}_{mkt}$

R(7d)

7 日对数收益率



(7-day Log Return)

Binance API



GET /api/v3/klines

symbol: BTCUSDT/ETHUSDT



interval: 1d



返回: close

3,652 条

$R_t^{(7d)} = \ln(P_t / P_{t-7})$



R(30d)

30 日对数收益率



(30-day Log Return)

Binance API



GET /api/v3/klines

symbol: BTCUSDT/ETHUSDT



interval: 1d



返回: close

3,652 条

$R_t^{(30d)} = \ln(P_t / P_{t-30})$



RSI

14 周期相对强弱指标

Binance API



GET /api/v3/klines

基于日线 Close 价格计算

3,652 条

标准 14 日 RSI 公式，归一化映射：$(RSI - 50) / 30$



HV

20 日已实现波动率



(20d Historical Vol)

Binance API



GET /api/v3/klines

基于日对数收益率 $r_\tau = \ln(P_\tau / P_{\tau-1})$

3,652 条

$HV_t = \sqrt{\frac{365}{20} \sum_{\tau=t-19}^{t} (r_\tau - \bar{r})^2} \times 100\%$

微观结构子空间 $\mathcal{S}_{mic}$

FR

永续合约资金费率



(24h Mean Funding Rate)

Binance Futures API



GET /fapi/v1/fundingRate

symbol: BTCUSDT/ETHUSDT



返回每 8h 费率 fundingRate

3,652 条

$FR_t = \frac{1}{3} \sum_{k=1}^3 FR_{t,k}$



(每日 3 次 8 小时资金费率的算术平均)



ΔOI

24 小时未平仓量变化率

Deribit / Coinglass API

open_interest (全网或 Deribit+Binance 合计)

3,652 条

$\Delta OI_t = \frac{OI_t - OI_{t-1}}{OI_{t-1}} \times 100\%$



LS

大户多空持仓人数/头寸比

Binance Futures API



GET /futures/data/topLongShortPositionRatio

symbol: BTCUSDT/ETHUSDT



period: 1d

3,652 条

直接取 longShortRatio (多头持仓量 / 空头持仓量)

链上资金流子空间 $\mathcal{S}_{flow}$

NetFlow

24 小时交易所净流入量

Coinglass API 或 Glassnode API

exchange_net_flow (单位: BTC / ETH)

3,652 条

$NetFlow_t = \text{Inflow}_t - \text{Outflow}_t$



(正值代表提币入场/卖压，负值代表流出提现)

🎰 三、 期权策略回放引擎（Replay Engine）专有数据

为支持 CBR 4R 循环中的 Reuse（策略回放） 与 Revise（参数修正），除了上述 12 维标量特征外，还必须抓取历史全量期权链日终快照（Option Chain Daily Snapshots）。

1. 每日期权链快照数据规范

数据源：Deribit API (public/get_book_summary_by_currency) 或 Tardis.dev / Amberdata 历史归档。

采样时点：每日 23:59:59 UTC。

数据总量：1,095,600 条记录（每天在售期权合约约 600 个 × 1,826 天）。

2. 必须包含的字段明细

字段名 (Field)

数据类型

示例值

业务用途 / 论文对应模块

timestamp

Int64 (ms)

1704067199000

时间戳对齐

instrument_name

String

BTC-29MAR24-60000-C

期权合约唯一标识（标的-到期日-行权价-类型）

underlying_price

Float64

42250.50

标的价格 $S_0$，用于计算 ATM/OTM 状态与 Delta 匹配

strike

Float64

60000.00

行权价 $K$，用于参数修正（Revise）模块

expiration_timestamp

Int64 (ms)

1711708800000

计算剩余到期天数 $DTE = (T_{exp} - t) / 86400$

option_type

String

"call" / "put"

期权方向

mark_price

Float64

0.0235 (BTC单位)

结算标记价，用于每日盯市（Mark-to-Market）计算盈亏

bid_price / ask_price

Float64

0.0230 / 0.0240

买卖盘口价，用于精确计算滑点与交易成本

mark_iv

Float64

55.42 (%)

标记隐含波动率 $\sigma_t$，用于 BS 模型拟合

delta

Float64

0.2451

单腿希腊字母 $\Delta$，用于 $\Delta$-neutral 对冲与动态匹配

gamma

Float64

0.00003

凸性风险指标 $\Gamma$

vega

Float64

12.45

波动率敏感度 $\mathcal{V}$

theta

Float64

-15.20

时间衰减速率 $\Theta$

volume

Float64

124.5

当日成交量，过滤无流动性的“僵尸合约”

open_interest

Float64

1520.0

未平仓量，评估撮合可行性

🛠️ 四、 数据抓取与 API 调用操作指南

推荐编写 Python 脚本 (scripts/fetch_all_data.py) 分三步异步并发抓取：

1. Deribit DVOL 指数抓取代码示例

import requests
import pandas as pd

def fetch_deribit_dvol(currency="BTC", start_timestamp=1609459200, end_timestamp=1767225600):
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    params = {
        "currency": currency,
        "start_timestamp": start_timestamp * 1000, # ms
        "end_timestamp": end_timestamp * 1000,
        "resolution": "1D"
    }
    response = requests.get(url, params=params).json()
    data = response['result']['data']
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


2. Binance 现货 K 线与资金费率抓取代码示例

def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_str="1 Jan, 2021", end_str="31 Dec, 2025"):
    # 使用 python-binance 或 requests 调用 GET /api/v3/klines
    # 提取 close 价格序列以计算 R(7d), R(30d), RSI, HV
    pass


📋 五、 数据清洗、预处理与安全校验 CheckList

在将抓取到的原始数据存入案例库之前，必须通过以下 4 项校验：

[ ] 时间戳无缝对齐（UTC 24:00 Alignment）：
所有衍生品（Deribit）、现货（Binance）及链上数据（Coinglass）统一裁切并对齐至 UTC 23:59:59。

[ ] 严格防范未来函数（Look-Ahead Bias Prevention）：
在 $t$ 时刻构造向量 $V_t$ 时，严禁引入任何 $t+1$ 及以后的数据。$HV^{(20d)}$、$R^{(7d)}$、$IVP$ 的计算窗口必须严格向后看（Historical Window Only）。

[ ] 缺失值补全规则（Missing Value Imputation）：

若某日缺失率 $< 5\%$：采用前值填充（Forward Fill / LOCF）。

若某日缺失率 $> 5\%$：阻断报错并触发报警，人工检查交易所 API 停机维护日志。

[ ] 异常值截断（Outlier Clipping）：
特征归一化时，所有特征经 $f_{norm}$ 映射后，通过 np.clip(V, -1.0, 1.0) 强制截断至 $[-1, +1]$ 空间，防止极端黑天鹅插针破坏马氏距离矩阵的协方差计算。