# House Prices：训练、验证与 Kaggle 提交

Python 3.12+。从仓库根目录运行 `uv sync --frozen` 安装依赖。数据放在
`data/raw/{train,test,sample_submission}.csv`。比赛主流程直接读取 CSV，
复用原有结构清洗，不需要 PostgreSQL 或 `.env`。

```powershell
# 先生成基线；加 --upload 可直接上传 Kaggle（需要有效登录及已接受比赛规则）
uv run python -m src.submit --output reports/submission_baseline.csv --upload
# 开发集上的模型选择：3 次 × 5 折，运行需要数分钟
uv run python -m src.train
# 冻结模型后使用独立 holdout 一次，并用全部 1460 行重训
uv run python -m src.finalize
# 最终提交（生成 CSV 默认不联网）
uv run python -m src.submit --artifact models/house_price_final.joblib --output reports/submission.csv
uv run python -m src.predict
uv run ruff check src competition tests
uv run pytest -q
```

当前已验证基线 Public LB **0.12633**（提交 **57107857**）。Windows 执行账号
只能读取 OAuth 凭据时，可以在进程内刷新令牌，不修改凭据文件：

```powershell
uv run python -m src.leaderboard --file reports/submission.csv --upload --memory-auth
# 查询真实分数；不带 --upload 不会发起新提交
uv run python -m src.leaderboard --file reports/submission.csv --memory-auth
uv run python -m src.report
```

该命令按 CSV 的 SHA256 匹配最近提交，避免重复上传同一结果，分数写入
`*.leaderboard.json`。仅使用现有登录，不在仓库保存令牌。

## 评估约定

- 训练目标为自然对数 `log(SalePrice)`，预测取 `exp`；主指标为
  `sqrt(mean((log(y)-log(pred))**2))`。不应用会改变比赛指标的 smearing。
- 保持原来的 80/20 split（seed=42）。1168 行开发集用于模型选择，292 行
  holdout 仅在选择冻结后使用。全量基线虽然提前训练用于提交，但不用于 holdout 评估。
- 所有候选共用 seed 42/52/62 的 5 折；折内预处理。报告同时列出折均值、
  折标准差及 pooled repeated log RMSE；折标准差不是均值的标准误。
- 配对检验使用 corrected resampled t 近似：
  `SE = sqrt((1/n_folds + n_valid/n_train) * var(paired_differences))`，
  全部模型对采用 Holm 校正；95% CI 为单个比较的区间。
  重复折不是独立样本，这仍是近似推断，不能把未显著解释为等价。
- 预先固定保守选择规则：只有经 Holm 校正显著优于 CatBoost 基线的候选
  才可替换基线；多个满足条件时按平均 log RMSE 选择。LB 不参与模型选择。
- 高价段阈值只由每个训练折的目标 90% 分位数计算；评估报告另列 >$400k。
  真实价格仅用于误差分组，不进入特征或预测路由。

## 特征和模型范围

默认候选仅保留 CatBoost、XGBoost、Gradient Boosting。移除原来默认比较中的
单棵树、原价线性模型、随机森林、HistGB 和 LightGBM；这是收窄实验预算，
不代表证明这些模型在所有配置下都差。Dummy 仅作 `src.baseline` 的参考。

CatBoost 做 baseline / 面积与房龄 / 交易条件交互三组消融。新增 Normal、
Partial、Abnorml 标志，以及与质量、居住面积、总面积、房龄的交互，另加
质量×面积和装修年龄。特征只使用单行已知输入，不做全量目标编码。
原始 `sale_condition` 本来就已输入 CatBoost，新增交互是否有效由 CV 决定。

## 交付文件

- `reports/repeated_cv/`：同折分数、逐房屋逐次 OOF、配对检验、分段误差、
  固定 ID 划分、选择配置。
- `reports/model_comparison.csv`：当前统一 log 指标排名。
- `reports/final_holdout.json` / `holdout_predictions.csv`：最终独立评估。
- `models/house_price_final.joblib`：含原始特征顺序、变换版本、目标空间的全量模型。
- `reports/submission.csv` 及同名 JSON：提交文件、SHA256、训练元数据与提交状态。

`src.finalize` 发现已有 holdout 报告时拒绝重跑，防止反复调参污染验证。
如只需恢复全量模型，可用 `src.submit --selection reports/repeated_cv/selection.json`。
joblib 仅加载自己生成的可信文件。

历史 `competition/oof.py`、`log_target_oof.py` 和旧特征实验保留供复现，不是当前
模型选择入口。`src.ingest` / `src.clean` 为可选 PostgreSQL 流程；配置见
`.env.example`。根目录 melon 草稿不属于应用或 CI 范围。原有用户修改予以保留。

真实排行榜分数必须来自 Kaggle，CV/holdout 不能代替 LB。提交失败时 JSON
记录失败状态，不会填造分数。详细运行结论见 `reports/FINAL_REPORT.md`。
