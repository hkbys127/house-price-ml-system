# 项目改进评审报告

评审对象：`house-price-ml-system`（Kaggle House Prices，约 2,100 行自有 Python 代码）
评审方式：通读全部源码 + 静态检查（ruff）+ 数据/结果文件量化核查 + 一组对照实验复现
关键基线：`catboost_log_target` 的 OOF log RMSE = **0.1230**（log 空间，即比赛指标），美元 RMSE = $26,414

---

## 一、总体评价

这个项目的**方法论骨架是对的**，比多数同阶段的 Kaggle 项目更规范：

- 分层清晰：`ingest → clean → features → train/evaluate → predict`，职责边界明确。
- `clean.py` 明确只做「不依赖全量统计量」的结构性清洗（docstring 里写明了），避免了最典型的泄漏陷阱。
- 预留 holdout、且**不拿 holdout 做模型选择**——这是正确的纪律。
- 所有模型共用同一套 fold（`competition/oof.py` 里显式复用同一个 `splits`），OOF 口径一致，可比。
- `competition/feature_stability.py` 主动做多 seed（42/52/62）稳定性验证——这个直觉很好。
- `sql/02_quality_checks.sql`、`03_clean_checks.sql` 的质检写得相当扎实（ID 唯一性、目标合法性、缺失分布、异常值探查）。
- 代码有类型注解、`from __future__ import annotations`，ruff 对 `src/` 和 `competition/` 基本零告警。

**最大的问题不是「代码写得不好」，而是「项目缺了最后一公里，而且当前的评估口径不足以支撑它已经做出的结论」。**

三个最关键的判断：

1. **项目产不出 `submission.csv`。** 全仓库 `*.py` 里搜不到任何 `submission` 相关代码。有 `sample_submission.csv`、有 `predict.py`，但没有「跑 test.csv → 生成提交文件」的路径。作为 Kaggle 项目，核心交付物缺失。
2. **模型排名的差异淹没在噪声里，但代码把结论当成了确定的。** `model_comparison.csv` 前三名 CV RMSE 只差 **$457**，而 `cv_rmse_std` 是 **$4,921 ~ $6,367**。前六名在统计上无法区分。然而 `train_candidate.py` 已经据此把 CatBoost 硬编码成唯一候选。
3. **真正的改进空间在误差结构，不在换模型。** 残差与真实房价的相关系数 **−0.4442**（严重的收缩/均值回归偏差），而**仅第 523 行一行就占总平方误差的 12.80%，前 20 行占 56.12%**。指标被极少数样本主导，这既解释了为什么模型排名不稳定，也指明了该往哪里使劲。

---

## P0 — 最高价值改进

### P0-1 补齐最后一步：生成 Kaggle 提交文件

**现状**：`grep -r "submission" --include=*.py` → 无匹配。`src/predict.py` 只能对单条手写特征字典预测，无法批量跑 `test.csv`。

**影响**：项目目前无法参赛、无法验证真实 LB 分数，前面所有 OOF 工作都无法闭环。

**建议**：新增 `src/submit.py`，复用已训练的 artifact 批量推理。注意三个易错点：列顺序（`artifact["feature_columns"]`）、类别列缺失填补、以及输出必须包含 `Id` 且行数等于 1459。

```python
# src/submit.py（骨架）
import joblib
import pandas as pd

from src.catboost_utils import prepare_catboost_data
from src.split import load_training_data  # 仅为取得列口径，实际读 test 表

def main() -> None:
    artifact = joblib.load(MODEL_PATH)
    test = pd.read_sql_table("house_test_clean", con=get_engine())
    test_ids = test["id"]
    X = test[artifact["feature_columns"]]
    X = prepare_catboost_data(X, artifact["categorical_columns"])
    preds = artifact["model"].predict(X)
    # 若模型是 log 目标训练，这里必须 exp() 回来 + 偏置校正（见 P0-4）
    pd.DataFrame({"Id": test_ids, "SalePrice": preds}).to_csv(
        BASE_DIR / "reports" / "submission.csv", index=False
    )
```

### P0-2 让模型选择具备统计意义

**现状问题**（证据来自 `reports/model_comparison.csv`）：

| 现象 | 数值 |
|---|---|
| 前 3 名 CV RMSE 差距 | $457（1.7%）|
| 单模型折间标准差 `cv_rmse_std` | $4,921 – $6,367 |
| CatBoost vs XGBoost 拟合耗时 | 21.65s vs 0.82s（**26 倍**）换 $457 |

用**单次 5 折 CV** 在同一样本上比较十几个模型，然后按小数点后几位排序，本质上是在拟合噪声。而且排名的目标量是**美元 RMSE，而比赛指标是 log 空间 RMSE**——`model_comparison.csv` 里根本没有 log 空间的列，选择目标与优化目标不一致。

**建议**：

1. `model_comparison.csv` 增加 `cv_log_rmse_mean` / `cv_log_rmse_std`，**用比赛指标排序**。`competition/oof.py` 里已经有 `competition_log_rmse()`，直接复用即可。
2. 引入**重复交叉验证**（`RepeatedKFold`，≥3 个 seed），报告 `mean ± std`，并对候选项做**配对检验**（paired bootstrap 或 paired t-test on per-fold/per-row errors）。只有差异大于噪声才写进结论。
3. 砍掉明显无价值的对照：`decision_tree_full`（train RMSE = 0.0，纯记忆）、`decision_tree_depth5`、`linear_regression`、`ridge`/`lasso`（≈$31.6k）、`dummy_median`（只该留在 `baseline.py`）。
4. 报告口径要写清：现在的 `cv_rmse_mean` 是**5 折 RMSE 的算术平均，不是 pooled RMSE**（因 Jensen 不等式，pooled 值更高，例如 gradient_boosting 27,129 vs 27,876）。两者不能混用。

### P0-3 修正系统性收缩偏差（真正的精度空间）

**证据**（由 `reports/catboost_oof_error_analysis.csv` 计算，`residual = pred − true`）：

| 分组 | n | 平均残差（pred − true）| 说明 |
|---|---|---|---|
| 最低十分位（$34.9k–107k）| 118 | **+$10,130** | 明显高估 |
| 第 6–7 十分位（$165k–200k）| 235 | 约 +$1,200 | 基本无偏 |
| 最高十分位（$277k–745k）| 117 | **−$25,482** | 明显低估 |
| > $400k | 21 | **−$73,364** | 严重低估 |

`corr(y_true, residual) = −0.4442`；把残差对真实房价做 OLS，斜率 **−0.1516**——即真实房价每高 $1，预测只跟上约 $0.85。这是典型的向均值收缩。

**建议**（按性价比排序）：

1. **坚持 log 目标**。SalePrice 原始偏度 1.8829，`log1p` 后仅 0.1213。log 空间建模本身就缓解收缩，这也是 log-target 实验在 CatBoost 上有效的根因（配对 bootstrap：Δ = −0.00309，95% CI [−0.00565, −0.00051]，P(improve)=0.990）。
2. **分段/分位数建模**：对高价段单独建模，或加入「价格段」作为特征（注意用训练折内分位数，避免泄漏）。
3. **考虑目标编码 + 高价样本加权**，直接对冲高估低价、低估高价的对称误差。
4. **`sale_condition` 是强信号但没被充分利用**：`Partial` MAE $28,755（n=98）、`Abnorml` MAE $22,073（n=77），而 `Normal` 只有 $12,995（n=964）——非正常交易条件的误差是正常交易的 **2.2–2.6 倍**。至少应加 `is_normal_sale` 之类的显式特征。

### P0-4 补上 log 目标的偏置校正（近乎免费的收益）

**证据**：`competition/log_target_oof.py:89,168` 直接 `np.exp(log_predictions)`。模型预测的是 `E[log(price)]`，`exp()` 后得到的是**几何均值**而非算术均值，因此美元空间系统性偏低约 1%（`mean(pred)/mean(true)` = 0.9876–0.9906）。

**影响范围（已量化）**：log 空间的比赛指标**不受影响**（`log(exp(x)) == x`），但美元 RMSE 白丢钱：

| 模型 | 当前美元 RMSE | 加 Duan smearing 后 |
|---|---|---|
| catboost | $26,414 | **$26,324** |
| xgboost | $26,903 | **$26,741** |
| gradient_boosting | $28,492 | **$28,382** |

**建议**：在管道内（而非事后）加 Duan smearing 因子 `mean(exp(residual_log))` 或 `exp(σ²/2)`。注意：如果最终评分只看 log 指标，这项可以不急；但只要报表里有美元 RMSE，就该修。

---

## P1 — 正确性与可维护性

### P1-1 `prepare_catboost_data` 有三份实现，且常量被绕过

| 位置 | 是否使用常量 |
|---|---|
| `src/catboost_utils.py:8` | ✅ `MISSING_CATEGORY_TOKEN` |
| `src/evaluate.py:101` | ❌ 硬编码 `"__MISSING__"` |
| `competition/oof.py:171` | ❌ 硬编码 `"__MISSING__"` |

三份都是**活代码**（分别被 `evaluate_catboost`、`run_catboost_oof` 调用），而 `competition/log_target_oof.py:11` 又去 import `src.catboost_utils` 的版本。一旦有人改了 `MISSING_CATEGORY_TOKEN`，另外两处会静默不同步，造成**训练/推理类别编码不一致**——这是不会报错的正确性 bug。

**建议**：删除 `src/evaluate.py` 和 `competition/oof.py` 里的副本，统一 import `src.catboost_utils`。同理，`ID_COLUMN` / `TARGET_COLUMN` 在 `src/clean.py:15-16` 和 `src/features.py:9-10` 各定义一份，应收敛到单一模块（例如 `src/schema.py`）。

### P1-2 语义型缺失值处理不完整（`ABSENCE_CATEGORY_COLUMNS` 漏项）

`src/clean.py:21-36` 的列表漏了两个关键列，二者都是「NA 表示不存在」而非「未知」：

**① `mas_vnr_type`（外墙饰面类型）**——训练集 872/1460 = **59.7%** 为 NaN（`X_dev` 中 683/1168）。数据字典 `data_description.txt:242-248` 明确列出 `None` 类别；实测这批行的 `MasVnrArea` 均值仅 **1.09**，而有值的组是 239–259。即 **NaN = 没有外墙饰面**。

- 对 CatBoost 路径：`prepare_catboost_data` 会把 NaN 变成独立的 `__MISSING__` 类别，**问题不大**（我的对照实验已证实改动无效，见附一）。
- 对 sklearn 管道（`ridge`/`lasso`/`linear_regression`，走 `src/features.py` 的 `SimpleImputer(strategy="most_frequent")`）：683 行会被填成 **`BrkFace`**——等于给 60% 的房子凭空安上「砖砌外墙」。这是**事实性错误**，不只是缺失值策略问题。

**② `garage_yr_blt`（车库建造年份）**——NaN 恰好 81 行，且与 `garage_type` 为 NaN 的 81 行**完全一致**（crosstab 验证：isna=True 的 81 行全部落在 GarageType 缺失上），即 **NaN = 没有车库**。当前 `SimpleImputer(median)` 会填成 **1980**，给 81 套无车库的房子编造出车库年份。

**建议**：`mas_vnr_type` 加入 `ABSENCE_CATEGORY_COLUMNS`；`garage_yr_blt` 缺失填 0（或改为 `has_garage` 布尔 + `garage_age` 派生特征）。顺带一提，`sql/03_clean_checks.sql:88` 已经把它命名为 `unknown_or_absent_garage_year`——作者其实已经意识到了「absent」语义，只是清洗逻辑没跟上。

### P1-3 `error_analysis.py` 重新推导 split，存在静默错配风险

`competition/error_analysis.py:68-73` 重新调用 `load_training_data()` + `make_holdout_split()` 来取特征，然后靠 `row_index` 做 `merge(how="left", validate="one_to_one")`。

`validate="one_to_one"` **只能捕获重复键，捕获不了「split 变了导致贴错特征」**。一旦改动 `HOLDOUT_SIZE` 或 `RANDOM_STATE`，脚本不会报错，只会把**错误的 neighborhood / overall_qual 贴到预测上**，产出一份看起来正常但结论错误的误差报告。

**建议**：不要重新推导。`reports/oof_log_target_predictions.csv` 里已经存了 `row_index`，直接用它 join 清洗表即可；或者把 split 落盘成 `reports/split_assignment.csv` 作为唯一事实来源。

### P1-4 holdout 永远处在「未使用」状态

`grep` 结果：只有 `src/baseline.py:89` 真正用了 holdout，而它评估的是 **dummy 模型**。`src/train.py:25` 和 `competition/oof.py:262` 只 print 了行数；`src/train_candidate.py:27` 直接 `X_dev, _, y_dev, _` 把 holdout 丢掉了。

因此 `models/house_price_catboost_dev_metadata.json` 里那句 `"Final holdout has not been used"` **永远不会变成 used**——292 行数据（20%）被永久冻结且无人消费。

**建议**：二选一。要么加 `src/finalize.py`：在模型选型冻结后，用 `X_dev + X_holdout` 全量重训、并在 holdout 上报告唯一一次最终分数（这才是 holdout 的正确用法）；要么承认没有独立验证需求，改为纯 CV 流程，不要留 20% 数据不用。

### P1-5 无法用 `python src/xxx.py` 运行，所有 `__main__` 守卫形同虚设

```
$ python src/split.py
ModuleNotFoundError: No module named 'src'

$ python -m src.split      # ✅ 唯一可行方式
X_dev shape:     (1168, 79)
```

9 个入口文件全都写了 `if __name__ == "__main__": main()`，但直接运行一律失败（因为它们都 import `src.*`）。这对使用者和后续维护都是绊脚石。

**建议**：在 `pyproject.toml` 加 CLI 入口，例如
```toml
[project.scripts]
hps-ingest = "src.ingest:main"
hps-clean  = "src.clean:main"
hps-train  = "src.train:main"
hps-submit = "src.submit:main"
```
这样 `uv run hps-train` 即可，README 也能写得清爽。

### P1-6 零测试

`tests/` 目录存在但**完全为空**，`pytest --collect-only` → `no tests collected`，而 pytest 已声明为 dev 依赖。

这个项目里最适合单测的恰恰是最容易悄悄错的部分：

- `to_snake_case`（列名映射，错了整条管道都错）
- `clean_dataframe` 的各类校验分支（重复 ID、目标 ≤ 0、test 集误含目标变量）
- `fill_semantic_absence`（P1-2 那个漏项本该被这条测试拦住）
- `competition_log_rmse` 与 `calculate_metrics` 的符号/空间口径
- `prepare_catboost_data` 的列顺序与类别一致性

**建议**：先补 10 个左右纯函数级测试即可获得大部分收益；`tests/` 需要一个 `conftest.py` 或 pytest 配置来把仓库根加入 `sys.path`。

---

## P2 — 工程与仓库卫生

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | `catboost_info/` 被提交且**每次训练都变脏** | `git status` 显示 4 个文件 modified，另有未跟踪的 `catboost_info/tmp/` | 加入 `.gitignore`；或设 `train_dir` 到临时目录 |
| 2 | 根目录 5 个 `melon*.py` 草稿被提交，且**不是合法 Python** | `melon-split.py:16` 缺冒号、`TARGET_COlUMN` 拼写错误、`from xgboost import data`；ruff 报 5×F821 + invalid-syntax | 删除或移入 `scratch/`（并 gitignore） |
| 3 | `README.md` **0 字节**；`pyproject.toml` description 仍是脚手架默认的 `"Add your description here"`；`main.py` 是 uv 默认的 Hello | 直接读文件 | 写清「如何跑通全流程」+ 当前最佳分数 0.1230 |
| 4 | 原始数据提交进 git | `data/raw/train.csv` 460KB、`test.csv` 451KB、`.zip` 203KB（与解压后的 csv 重复）| 数据不入库；保留 `.zip` 或提供下载脚本 |
| 5 | `api/`、`tests/`、`notebooks/` 三个**空目录** | 各 0 个条目；git 不跟踪空目录，克隆后并不存在 | 要么落地内容，要么删掉，避免误导 |
| 6 | `sql/` 缺 `01_`；compose 里的挂载是**无效的** | 只有 `02_`、`03_`；`compose.yaml` 挂 `./sql:/sql`，但 PostgreSQL 只在 `/docker-entrypoint-initdb.d` 自动执行 | 改挂载到 `/docker-entrypoint-initdb.d`，并补 `01_`（或重新编号） |
| 7 | 无 CI、无 pre-commit、无 `Dockerfile`、无 `.env.example` | 全部 `Test-Path` = False | 至少加一个跑 `ruff check` + `pytest` 的 GitHub Actions |
| 8 | 日志全是 `print`（62 处），无 `logging`，无实验追踪 | `grep print(` → 62 处，`grep logging` → 0 | 引入 `logging`；结果 CSV 增加 run 时间戳/ git commit / 超参数，否则无法回溯 |
| 9 | **完全没有超参数调优，也没有早停** | `grep -E "GridSearch\|RandomizedSearch\|optuna\|early_stopping\|eval_set\|od_type"` → **无匹配** | 加 `early_stopping_rounds` + 一个 `eval_set` 几乎必然改善过拟合；再做一轮 Optuna |
| 10 | 过拟合严重 | `rmse_gap`：lightgbm 23,020（train×5.0）、xgboost 18,886、catboost 14,645；`decision_tree_full` train RMSE **恰好 0.0** | 全部模型加正则/早停；固定 500 轮次是主因 |
| 11 | `StandardScaler` 对树模型无意义 | `src/features.py:54-65` 所有模型共用一个 preprocessor | 按模型族拆分：树模型只补缺失，线性模型才标准化 |
| 12 | `evaluate_catboost` 与 `evaluate_model` 大量重复 | `src/evaluate.py:44-98` vs `123-217` | 抽出统一的 CV + 指标计算函数 |
| 13 | `import time` 写在函数体内 | `src/evaluate.py:147` | 提到模块顶部 |
| 14 | 关键证据未落盘 | `competition/feature_stability.py:128-155` 算了跨 seed 汇总但**只 print，从不写 CSV** | 写入 `reports/feature_stability.csv` |
| 15 | 特征工程结论处在噪声内 | `feature_experiment_results.csv`：最佳 delta = 0.00024，而折间 std = 0.0136–0.0152（约 1/60）；且 `add_total_sf` 单独使用反而变差 +0.00115 | 结论要标注「不显著」；`total_sf`/`house_age` 的收益需多样本重复验证 |
| 16 | 列名可读性差 | `to_snake_case` 产出 `m_s_sub_class`、`m_s_zoning`、`bsmt_fin_s_f1`、`1st_flr_s_f` | 加缩写映射表（MS→ms、SF→sf、YrBlt→yr_blt）|
| 17 | 包结构不一致 | `src/` 有 `__init__.py`，`competition/` 没有 | 统一（或都改成 namespace package）|
| 18 | `.env` 弱口令，且无 `.env.example` | `POSTGRES_PASSWORD=ml_password` | 提供 `.env.example`；本地开发可接受但应注明 |

---

## 附一：我做的对照实验（用于验证 P0-3 / P1-2 的假设）

为验证「删掉 2 个 partial-sale 异常值是否能提升分数」，我用项目自身的模块（`src.catboost_utils` / `src.features` / `src.models` / `src.split`）和**完全相同的 fold（KFold, 5 折, seed=42）** 复现了 log-target CatBoost 管道：

- 先验证复现正确性：我的 as-is 变体 log RMSE = **0.122965**，与 `feature_experiment_results.csv` 中 `baseline` 的 `0.1229648792970072` **完全一致**，说明口径可对。
- 异常值用 `(gr_liv_area > 4000) & (sale_price < 300000)` 定位，命中 2 行：**position 523**（Edwards, 4,676 sqft, $184,750）与 **position 1298**（Edwards, 5,642 sqft, $160,000）。

| 变体 | log RMSE（全 dev）| log RMSE（排除这 2 行）| 美元 RMSE | 折间 std |
|---|---|---|---|---|
| A 现状 | **0.122965** | 0.117265 | $26,414 | 0.0136 |
| B 仅修语义缺失值 | 0.122965 | 0.117265 | $26,414 | 0.0136 |
| C 仅从**训练集**删这 2 行 | 0.127457 | 0.117827 | $28,442 | 0.0175 |
| D 两者都做 | 0.127457 | 0.117827 | $28,442 | 0.0175 |

**两个结论，其中一个是反直觉的：**

1. **「删掉那 2 个著名异常值」在这个项目里并不会提升 OOF 分数，反而更差**（0.1275 vs 0.1230，美元 RMSE 差 $2,028）。在排除这 2 行的口径下两者也基本持平（0.1173 vs 0.1178）。原因：只删掉 1168 行中的 2 行（0.17%）对模型影响很小，但模型再也学不到「超大户型在 Edwards 可能低价成交」这一模式，于是对这 2 行的预测更差——而恰恰是这 2 行主导了指标。所以 `sql/03_clean_checks.sql:116` 那个 `gr_liv_area > 4000` 的探查**直觉合理但实测不划算**，不要盲目照搬网上的「删异常值」套路。
2. **语义缺失值修复对 CatBoost 路径零影响**（B 与 A 逐位相同）。这符合预期：`prepare_catboost_data` 已把类别型 NaN 转成独立的 `__MISSING__`，CatBoost 也原生处理数值 NaN。**所以 P1-2 的修复对象是 sklearn 管道（ridge/lasso/linear），不是 CatBoost**——它是正确性问题，不是提分手段。这一点必须说清楚，避免被误当成调优技巧。

**顺带量化了指标的脆弱程度**：仅 position 523 一行就占总平方误差的 **12.80%**，前 5 行 34.62%，前 20 行 **56.12%**。把 2 个异常值从**评估集**排除后，log RMSE 从 0.1230 降到 **0.1173**。

这解释了两件事：为什么前六名模型在统计上无法区分（P0-2），以及为什么**当前指标不适合做细粒度模型选择**。

> 附注：我第一版实验脚本有 bug——用 `probe.index`（`read_sql` 返回的全新 0-based 索引）当位置，结果删错了行（删了 0、1 两行）。已修正为从同一个 `X_all/y_all` 派生掩码，上表是修正后的结果。项目里 `error_analysis.py` 的 join 有同类隐患，正是 P1-3 想说的风险。

---

## 附二：建议的执行顺序

| 阶段 | 动作 | 预期收益 |
|---|---|---|
| 1 | P0-1 补 `src/submit.py`，先拿到真实 LB 分数 | 闭环，得到唯一的真值参考 |
| 2 | P0-2 用 log RMSE 排序 + 重复 CV + 配对检验；砍掉无用模型 | 结论可信，省掉 26× 的无谓拟合时间 |
| 3 | P1-1 / P1-2 / P1-3 修正确性问题 | 消除静默错配与编码不一致 |
| 4 | P0-3 + P2-9 针对收缩偏差做特征/分段建模，并加早停与超参搜索 | 真正的精度提升空间所在 |
| 5 | P1-4 决定 holdout 的去留；P1-6 补测试 | 流程自洽、可回归 |
| 6 | P2 其余卫生项（.gitignore、README、删草稿、CI）| 可维护性 |

**如果只做三件事**：补 `submit.py`（P0-1）、把评估口径换成 log RMSE + 重复 CV + 配对检验（P0-2）、针对高价段收缩偏差做特征工程（P0-3）。
