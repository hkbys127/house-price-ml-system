# 项目收尾报告

真实 Public LB：**0.12633**；Kaggle 提交 **57107857**。
最终模型：**catboost_baseline**，全量训练 1460 行，提交 1459 行。

## 重复交叉验证

1168 行开发集，seed 42/52/62，各 5 折；候选共享相同划分。
折标准差用于描述波动，不是标准误；pooled 指标将全部重复验证残差合并。

| 模型 | 折均值 log RMSE | 折标准差 | pooled log RMSE | 单折训练秒数 |
|---|---:|---:|---:|---:|
| catboost_sale_interactions | 0.121351 | 0.011871 | 0.121888 | 19.40 |
| catboost_size_age | 0.123795 | 0.013257 | 0.124455 | 18.04 |
| catboost_baseline | 0.125111 | 0.013638 | 0.125798 | 17.93 |
| xgboost_baseline | 0.126158 | 0.016382 | 0.127147 | 0.40 |
| gradient_boosting_baseline | 0.128830 | 0.016907 | 0.129860 | 1.65 |

## 配对检验与选择

差值定义为 A−B；负值表示 A 更好。使用考虑训练折重叠的 corrected resampled t 近似，
10 个比较采用 Holm 校正。区间为单个比较的 95% CI，不是同时置信区间。

| A | B | Δ log RMSE | 95% CI | Holm p |
|---|---|---:|---|---:|
| catboost_baseline | catboost_sale_interactions | 0.003760 | [-0.001180, 0.008699] | 1.0000 |
| catboost_baseline | catboost_size_age | 0.001315 | [-0.002065, 0.004696] | 1.0000 |
| catboost_baseline | gradient_boosting_baseline | -0.003719 | [-0.011555, 0.004117] | 1.0000 |
| catboost_baseline | xgboost_baseline | -0.001048 | [-0.008645, 0.006550] | 1.0000 |
| catboost_sale_interactions | catboost_size_age | -0.002445 | [-0.006947, 0.002058] | 1.0000 |
| catboost_sale_interactions | gradient_boosting_baseline | -0.007479 | [-0.015157, 0.000199] | 0.5543 |
| catboost_sale_interactions | xgboost_baseline | -0.004807 | [-0.012790, 0.003175] | 1.0000 |
| catboost_size_age | gradient_boosting_baseline | -0.005035 | [-0.012937, 0.002868] | 1.0000 |
| catboost_size_age | xgboost_baseline | -0.002363 | [-0.009216, 0.004490] | 1.0000 |
| gradient_boosting_baseline | xgboost_baseline | 0.002672 | [-0.003486, 0.008829] | 1.0000 |

选择规则事先固定：仅当候选经校正显著优于 CatBoost 基线才替换；多个满足条件时取均值最低者。
不显著不等于模型等价，也不表示永远没有提升空间。未用 LB 或 holdout 进行选择。
按预定规则，最终采用 catboost_baseline。其余特征配置仅保留为实验，不进入本次提交。

## 高价段与交易条件特征

新增 Normal/Partial/Abnorml 标志、交易条件×质量/面积/房龄，以及质量×面积和装修年龄。
原始 sale_condition 已输入基线；新增交互的收益需由实测验证。以下每栋房重复验证 3 次，不能把三次当成三栋独立房屋。

| 模型 | 分段 | 独立房屋数 | MAE（美元） | 平均残差 pred−true（美元） | log RMSE |
|---|---|---:|---:|---:|---:|
| catboost_baseline | high_price_train_q90 | 124 | 38,561 | -27,727 | 0.156022 |
| catboost_baseline | over_400k | 21 | 85,524 | -83,158 | 0.252214 |
| catboost_baseline | sale_Abnorml | 77 | 22,282 | -2,928 | 0.219756 |
| catboost_baseline | sale_Normal | 964 | 13,168 | -1,473 | 0.102243 |
| catboost_baseline | sale_Partial | 98 | 30,002 | -4,900 | 0.190448 |
| catboost_sale_interactions | high_price_train_q90 | 124 | 35,247 | -25,345 | 0.146934 |
| catboost_sale_interactions | over_400k | 21 | 78,728 | -76,693 | 0.244341 |
| catboost_sale_interactions | sale_Abnorml | 77 | 22,397 | -3,943 | 0.215577 |
| catboost_sale_interactions | sale_Normal | 964 | 12,597 | -1,260 | 0.099553 |
| catboost_sale_interactions | sale_Partial | 98 | 27,939 | -4,325 | 0.175871 |

本次基线 Partial / Normal MAE 比为 **2.28**；Abnorml / Normal 为 **1.69**。
交易条件与较大误差相关，不足以证明加入标志必然提高精度。高价分组阈值取各训练折的目标 90% 分位数；
真实房价只参与评估分组，不进入推理特征。未做未经验证的价格抬升或 smearing。

## 最终独立验证

冻结选择后一次性评估 292 行：log RMSE **0.130328**，美元 RMSE **26,926.77**，MAE **15,654.80**。
随后以全量标签重训；holdout 报告保留，入口阻止重复评估。

## 交付与验证

- 统一 log 目标训练、版本化特征及预测反变换；移除默认实验中的冗余模型。
- OOF 按真实 Id 落盘；完整性检查确认每个候选、每个 seed 恰好覆盖所有开发集房屋。
- 提交列名、1459 行 ID 顺序、价格正值及有限值均通过校验。
- Kaggle 真实分数与最终 CSV SHA256 绑定；相同内容不重复上传。
- README、11 个关键测试、GitHub Actions 检查流程均已补齐；本地 Ruff、pytest 通过。
- 保留原有用户修改和历史实验；已禁止新训练写入 catboost_info。

源代码、数据与模型哈希、依赖版本和参数见 training_run_manifest.json。
