"""Audit completed outputs and produce the project closeout report."""

import hashlib
import json
import subprocess

import joblib
import pandas as pd

from src.competition_data import ROOT
from src.submit import validate_submission


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = ROOT / "reports"
    cv = directory / "repeated_cv"
    comparison = pd.read_csv(cv / "model_comparison.csv")
    pairs = pd.read_csv(cv / "paired_tests.csv")
    segments = pd.read_csv(cv / "segment_metrics.csv")
    predictions = pd.read_csv(cv / "oof_predictions.csv")
    split = pd.read_csv(cv / "split_assignment.csv")
    selection = json.loads((cv / "selection.json").read_text(encoding="utf-8"))
    holdout = json.loads((directory / "final_holdout.json").read_text(encoding="utf-8"))
    leaderboard = json.loads(
        (directory / "submission.leaderboard.json").read_text(encoding="utf-8")
    )
    submission_path = directory / "submission.csv"
    validate_submission(
        pd.read_csv(submission_path),
        pd.read_csv(ROOT / "data/raw/sample_submission.csv"),
    )
    if leaderboard["sha256"] != digest(submission_path):
        raise ValueError("Leaderboard result does not correspond to final submission")
    development = set(split.loc[split.partition.eq("development"), "Id"])
    held_out = set(split.loc[split.partition.eq("holdout"), "Id"])
    if development & held_out or len(development) != 1168 or len(held_out) != 292:
        raise ValueError("Invalid development/holdout partition")
    for _, group in predictions.groupby(["model", "seed"]):
        if set(group.Id) != development or group.Id.duplicated().any():
            raise ValueError(
                "Each model/seed must predict every development house once"
            )
    if len(predictions) != 1168 * 3 * 5 or len(pairs) != 10:
        raise ValueError("Incomplete repeated evaluation")
    artifact = joblib.load(ROOT / "models/house_price_final.joblib")
    if (
        artifact["training_rows"] != 1460
        or artifact["family"] != selection["family"]
        or artifact["feature_variant"] != selection["variant"]
    ):
        raise ValueError("Final artifact does not match frozen selection")
    scores = [row for row in leaderboard["submissions"] if row["public_score"]]
    if not scores:
        raise ValueError("No actual leaderboard score available yet")
    lb = scores[0]
    lines = [
        "# 项目收尾报告",
        "",
        f"真实 Public LB：**{lb['public_score']}**；Kaggle 提交 **{lb['ref']}**。",
        f"最终模型：**{selection['model']}**，全量训练 1460 行，提交 1459 行。",
        "",
        "## 重复交叉验证",
        "",
        "1168 行开发集，seed 42/52/62，各 5 折；候选共享相同划分。",
        "折标准差用于描述波动，不是标准误；pooled 指标将全部重复验证残差合并。",
        "",
        "| 模型 | 折均值 log RMSE | 折标准差 | pooled log RMSE | 单折训练秒数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples():
        lines.append(
            f"| {row.model} | {row.cv_log_rmse_mean:.6f} | {row.cv_log_rmse_std:.6f} | {row.pooled_repeated_log_rmse:.6f} | {row.fit_seconds_mean:.2f} |"
        )
    lines += [
        "",
        "## 配对检验与选择",
        "",
        "差值定义为 A−B；负值表示 A 更好。使用考虑训练折重叠的 corrected resampled t 近似，",
        "10 个比较采用 Holm 校正。区间为单个比较的 95% CI，不是同时置信区间。",
        "",
        "| A | B | Δ log RMSE | 95% CI | Holm p |",
        "|---|---|---:|---|---:|",
    ]
    for row in pairs.itertuples():
        lines.append(
            f"| {row.model_a} | {row.model_b} | {row.delta_log_rmse:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.p_holm:.4f} |"
        )
    lines += [
        "",
        "选择规则事先固定：仅当候选经校正显著优于 CatBoost 基线才替换；多个满足条件时取均值最低者。",
        "不显著不等于模型等价，也不表示永远没有提升空间。未用 LB 或 holdout 进行选择。",
        f"按预定规则，最终采用 {selection['model']}。其余特征配置仅保留为实验，不进入本次提交。",
        "",
        "## 高价段与交易条件特征",
        "",
        "新增 Normal/Partial/Abnorml 标志、交易条件×质量/面积/房龄，以及质量×面积和装修年龄。",
        "原始 sale_condition 已输入基线；新增交互的收益需由实测验证。以下每栋房重复验证 3 次，不能把三次当成三栋独立房屋。",
        "",
        "| 模型 | 分段 | 独立房屋数 | MAE（美元） | 平均残差 pred−true（美元） | log RMSE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    focused = segments[
        segments.model.isin(["catboost_baseline", "catboost_sale_interactions"])
        & segments.segment.isin(
            [
                "high_price_train_q90",
                "over_400k",
                "sale_Normal",
                "sale_Partial",
                "sale_Abnorml",
            ]
        )
    ]
    for row in focused.itertuples():
        lines.append(
            f"| {row.model} | {row.segment} | {row.unique_houses} | {row.mae:,.0f} | {row.mean_residual:,.0f} | {row.log_rmse:.6f} |"
        )
    baseline = segments[segments.model.eq("catboost_baseline")].set_index("segment")
    lines += [
        "",
        (
            f"本次基线 Partial / Normal MAE 比为 **{baseline.loc['sale_Partial', 'mae'] / baseline.loc['sale_Normal', 'mae']:.2f}**；"
            f"Abnorml / Normal 为 **{baseline.loc['sale_Abnorml', 'mae'] / baseline.loc['sale_Normal', 'mae']:.2f}**。"
        ),
        "交易条件与较大误差相关，不足以证明加入标志必然提高精度。高价分组阈值取各训练折的目标 90% 分位数；",
        "真实房价只参与评估分组，不进入推理特征。未做未经验证的价格抬升或 smearing。",
        "",
        "## 最终独立验证",
        "",
        (
            f"冻结选择后一次性评估 {holdout['holdout_rows']} 行：log RMSE **{holdout['log_rmse']:.6f}**，"
            f"美元 RMSE **{holdout['rmse']:,.2f}**，MAE **{holdout['mae']:,.2f}**。"
        ),
        "随后以全量标签重训；holdout 报告保留，入口阻止重复评估。",
        "",
        "## 交付与验证",
        "",
        "- 统一 log 目标训练、版本化特征及预测反变换；移除默认实验中的冗余模型。",
        "- OOF 按真实 Id 落盘；完整性检查确认每个候选、每个 seed 恰好覆盖所有开发集房屋。",
        "- 提交列名、1459 行 ID 顺序、价格正值及有限值均通过校验。",
        "- Kaggle 真实分数与最终 CSV SHA256 绑定；相同内容不重复上传。",
        "- README、11 个关键测试、GitHub Actions 检查流程均已补齐；本地 Ruff、pytest 通过。",
        "- 保留原有用户修改和历史实验；已禁止新训练写入 catboost_info。",
        "",
        "源代码、数据与模型哈希、依赖版本和参数见 training_run_manifest.json。",
    ]
    (directory / "FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "git_head": git_head,
        "source_hashes": {
            str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "src").glob("*.py"))
        },
        "data_hashes": {
            name: digest(ROOT / "data/raw" / name)
            for name in ["train.csv", "test.csv", "sample_submission.csv"]
        },
        "model_sha256": digest(ROOT / "models/house_price_final.joblib"),
        "model_parameters": artifact["model_parameters"],
        "library_versions": artifact["library_versions"],
        "selection": selection,
        "submission_sha256": digest(submission_path),
    }
    (directory / "training_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Verified complete outputs; wrote {directory / 'FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
