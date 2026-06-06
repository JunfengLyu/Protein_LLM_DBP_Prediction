#!/usr/bin/env python3
"""Train balanced binary DBP classifiers with ESM-2 and average-pooling features."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import seaborn as sns
except ImportError:
    sns = None


FEATURE_COLUMNS = {
    "Average pooling": "average_pooling_embedding",
    "ESM-2": "embedded_sequence",
}
MODEL_ORDER = ["LR", "RF", "SVM", "MLP"]
AMBIGUITY_MODELS = ["LR", "SVM"]
ROC_COLORS = {
    "LR": "#2B6CB0",
    "RF": "#2F855A",
    "SVM": "#B7791F",
    "MLP": "#9F1239",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DBP binary baseline models.")
    parser.add_argument(
        "--input",
        default="washed_data/human_protein_table_embedded.tsv",
        help="TSV containing protein_id, is_DBP, embedded_sequence, and average_pooling_embedding.",
    )
    parser.add_argument(
        "--outdir",
        default="Results/04_Model_training",
        help="Output directory for model-training tables and figures.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=9)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--mlp-max-iter", type=int, default=350)
    parser.add_argument(
        "--ambiguity-ratios",
        default="0,0.1,0.25,0.5,1.0",
        help="Comma-separated ratios of ambiguity samples added to each training fold.",
    )
    return parser.parse_args()


def prepare_output_dir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for child in outdir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_vector(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=float)


def vector_matrix(series: pd.Series) -> np.ndarray:
    return np.vstack([load_vector(value) for value in series])


def make_balanced_binary_dataset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    binary = df[df["is_DBP"].isin([0, 1])].copy()
    positives = binary[binary["is_DBP"] == 1]
    negatives = binary[binary["is_DBP"] == 0]
    n_pos = len(positives)
    if len(negatives) < n_pos:
        raise ValueError("Not enough negative samples for balanced sampling.")
    sampled_negatives = negatives.sample(n=n_pos, random_state=random_state, replace=False)
    balanced = pd.concat([positives, sampled_negatives], axis=0)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def make_models(random_state: int, rf_trees: int, mlp_max_iter: int) -> dict[str, object]:
    return {
        "LR": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "RF": RandomForestClassifier(
            n_estimators=rf_trees,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        alpha=1e-4,
                        early_stopping=True,
                        max_iter=mlp_max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def model_scores(model: object, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    raise ValueError("Model does not expose predict_proba or decision_function.")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "average_precision",
    ]
    rows = []
    grouped = fold_metrics.groupby(["feature", "model"], sort=False)
    for (feature, model), group in grouped:
        row = {"feature": feature, "model": model}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_ratios(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def mean_roc_with_ci(records: list[dict[str, object]], mean_fpr: np.ndarray) -> dict[str, np.ndarray | float]:
    tprs = []
    aucs = []
    for record in records:
        fpr, tpr, _ = roc_curve(record["y_true"], record["y_score"])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc_score(record["y_true"], record["y_score"]))
    tprs_array = np.asarray(tprs, dtype=float)
    aucs_array = np.asarray(aucs, dtype=float)
    n = len(records)
    tpr_ci = 1.96 * np.std(tprs_array, axis=0, ddof=1) / np.sqrt(n)
    auc_ci = 1.96 * np.std(aucs_array, ddof=1) / np.sqrt(n)
    return {
        "mean_tpr": np.mean(tprs_array, axis=0),
        "tpr_ci": tpr_ci,
        "mean_auc": float(np.mean(aucs_array)),
        "auc_ci": float(auc_ci),
    }


def mean_pr_with_ci(records: list[dict[str, object]], mean_recall: np.ndarray) -> dict[str, np.ndarray | float]:
    precisions = []
    aps = []
    for record in records:
        precision, recall, _ = precision_recall_curve(record["y_true"], record["y_score"])
        recall = recall[::-1]
        precision = precision[::-1]
        interp_precision = np.interp(mean_recall, recall, precision)
        interp_precision[0] = precision[0]
        precisions.append(interp_precision)
        aps.append(average_precision_score(record["y_true"], record["y_score"]))
    precisions_array = np.asarray(precisions, dtype=float)
    aps_array = np.asarray(aps, dtype=float)
    n = len(records)
    precision_ci = 1.96 * np.std(precisions_array, axis=0, ddof=1) / np.sqrt(n)
    ap_ci = 1.96 * np.std(aps_array, ddof=1) / np.sqrt(n)
    return {
        "mean_precision": np.mean(precisions_array, axis=0),
        "precision_ci": precision_ci,
        "mean_ap": float(np.mean(aps_array)),
        "ap_ci": float(ap_ci),
    }


def plot_roc_curves(roc_records: list[dict[str, object]], outdir: Path) -> None:
    mean_fpr = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)

    for ax, feature in zip(axes, FEATURE_COLUMNS):
        feature_records = [record for record in roc_records if record["feature"] == feature]
        for model_name in MODEL_ORDER:
            model_records = [record for record in feature_records if record["model"] == model_name]
            if not model_records:
                continue
            roc_stats = mean_roc_with_ci(model_records, mean_fpr)
            mean_tpr = roc_stats["mean_tpr"]
            tpr_ci = roc_stats["tpr_ci"]
            mean_auc = roc_stats["mean_auc"]
            auc_ci = roc_stats["auc_ci"]
            color = ROC_COLORS[model_name]
            ax.plot(
                mean_fpr,
                mean_tpr,
                color=color,
                linewidth=1.7,
                label=f"{model_name} AUC={mean_auc:.3f}±{auc_ci:.3f}",
            )
            ax.fill_between(
                mean_fpr,
                np.maximum(mean_tpr - tpr_ci, 0),
                np.minimum(mean_tpr + tpr_ci, 1),
                color=color,
                alpha=0.16,
                linewidth=0,
            )
        ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
        ax.set_title(feature)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.grid(color="#E6E6E6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, edgecolor="black", fontsize=7, loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "02_test_roc_auc_curves.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_panel(records: list[dict[str, object]], outdir: Path) -> None:
    mean_fpr = np.linspace(0, 1, 201)
    mean_recall = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 8.2), sharex=False, sharey=False)

    for col_idx, feature in enumerate(FEATURE_COLUMNS):
        feature_records = [record for record in records if record["feature"] == feature]
        ax_roc = axes[0, col_idx]
        ax_pr = axes[1, col_idx]

        for model_name in MODEL_ORDER:
            model_records = [record for record in feature_records if record["model"] == model_name]
            if not model_records:
                continue
            color = ROC_COLORS[model_name]

            roc_stats = mean_roc_with_ci(model_records, mean_fpr)
            mean_tpr = roc_stats["mean_tpr"]
            tpr_ci = roc_stats["tpr_ci"]
            ax_roc.plot(
                mean_fpr,
                mean_tpr,
                color=color,
                linewidth=1.7,
                label="{} AUC={:.3f}±{:.3f}".format(
                    model_name,
                    roc_stats["mean_auc"],
                    roc_stats["auc_ci"],
                ),
            )
            ax_roc.fill_between(
                mean_fpr,
                np.maximum(mean_tpr - tpr_ci, 0),
                np.minimum(mean_tpr + tpr_ci, 1),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

            pr_stats = mean_pr_with_ci(model_records, mean_recall)
            mean_precision = pr_stats["mean_precision"]
            precision_ci = pr_stats["precision_ci"]
            ax_pr.plot(
                mean_recall,
                mean_precision,
                color=color,
                linewidth=1.7,
                label="{} AP={:.3f}±{:.3f}".format(
                    model_name,
                    pr_stats["mean_ap"],
                    pr_stats["ap_ci"],
                ),
            )
            ax_pr.fill_between(
                mean_recall,
                np.maximum(mean_precision - precision_ci, 0),
                np.minimum(mean_precision + precision_ci, 1),
                color=color,
                alpha=0.16,
                linewidth=0,
            )

        ax_roc.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
        ax_roc.set_title(f"{feature} ROC")
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.legend(frameon=True, edgecolor="black", fontsize=6.6, loc="lower right")

        ax_pr.set_title(f"{feature} PR")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend(frameon=True, edgecolor="black", fontsize=6.6, loc="lower left")

        for ax in [ax_roc, ax_pr]:
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(color="#E6E6E6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / "06_roc_pr_auc_panel.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def confusion_cmap():
    if sns is not None:
        return sns.color_palette("mako", as_cmap=True)
    return LinearSegmentedColormap.from_list(
        "mako_like",
        ["#0B0405", "#241124", "#3B255A", "#3E4C8A", "#2F7797", "#2FA3A0", "#7FD0BA", "#DEF5E5"],
    )


def plot_confusion_matrices(records: list[dict[str, object]], outdir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 6.4), sharex=False, sharey=False)
    image = None

    for row_idx, feature in enumerate(FEATURE_COLUMNS):
        for col_idx, model_name in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            model_records = [
                record for record in records if record["feature"] == feature and record["model"] == model_name
            ]
            y_true = np.concatenate([record["y_true"] for record in model_records])
            y_pred = np.concatenate([record["y_pred"] for record in model_records])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
            cm_norm = cm / cm.sum(axis=1, keepdims=True)

            image = ax.imshow(cm_norm, cmap=confusion_cmap(), vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    value = cm_norm[i, j]
                    count = int(cm[i, j])
                    text_color = "black" if value > 0.72 else "white"
                    ax.text(j, i, f"{value:.2f}\n({count})", ha="center", va="center", color=text_color, fontsize=9)
            if row_idx == 0:
                ax.set_title(model_name, fontsize=12, pad=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["0", "1"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"])
            if row_idx == 1:
                ax.set_xlabel("Predicted")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel(f"{feature}\nTrue")
            else:
                ax.set_ylabel("")
            ax.tick_params(length=0)

    fig.subplots_adjust(left=0.08, right=0.90, top=0.88, bottom=0.10, wspace=0.25, hspace=0.36)
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.018, label="Row-normalized proportion")
    fig.savefig(outdir / "07_confusion_matrix_panel.png", dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def run_ambiguity_test(
    df: pd.DataFrame,
    train_valid: pd.DataFrame,
    test: pd.DataFrame,
    cv: StratifiedKFold,
    ratios: list[float],
    random_state: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    ambiguity = df[df["is_DBP"] == -1].copy().reset_index(drop=True)
    x_train_valid = vector_matrix(train_valid["embedded_sequence"])
    y_train_valid = train_valid["is_DBP"].astype(int).to_numpy()
    x_test = vector_matrix(test["embedded_sequence"])
    y_test = test["is_DBP"].astype(int).to_numpy()
    x_ambiguity = vector_matrix(ambiguity["embedded_sequence"])

    rows = []
    roc_records = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train_valid, y_train_valid), start=1):
        x_base = x_train_valid[train_idx]
        y_base = y_train_valid[train_idx]
        n_positive = int((y_base == 1).sum())
        for ratio in ratios:
            n_ambiguous = int(round(n_positive * ratio))
            if n_ambiguous > len(ambiguity):
                raise ValueError("Requested more ambiguity samples than available.")
            if n_ambiguous:
                rng = np.random.default_rng(random_state + fold_idx * 1000 + int(ratio * 1000))
                ambiguity_idx = rng.choice(len(ambiguity), size=n_ambiguous, replace=False)
                x_amb = x_ambiguity[ambiguity_idx]
            else:
                x_amb = np.empty((0, x_train_valid.shape[1]), dtype=float)

            for mode, assigned_label in [("negative-noise", 0), ("positive-noise", 1)]:
                if n_ambiguous:
                    x_train = np.vstack([x_base, x_amb])
                    y_train = np.concatenate([y_base, np.full(n_ambiguous, assigned_label, dtype=int)])
                else:
                    x_train = x_base
                    y_train = y_base

                models = make_models(random_state + fold_idx, rf_trees=300, mlp_max_iter=350)
                for model_name in AMBIGUITY_MODELS:
                    model = models[model_name]
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    y_score = model_scores(model, x_test)
                    metrics = evaluate_predictions(y_test, y_pred, y_score)
                    rows.append(
                        {
                            "feature": "ESM-2",
                            "model": model_name,
                            "mode": mode,
                            "ambiguity_ratio": ratio,
                            "fold": fold_idx,
                            "added_ambiguity_n": n_ambiguous,
                            "train_n0": int((y_train == 0).sum()),
                            "train_n1": int((y_train == 1).sum()),
                            "test_n0": int((y_test == 0).sum()),
                            "test_n1": int((y_test == 1).sum()),
                            **metrics,
                        }
                    )
                    roc_records.append(
                        {
                            "feature": "ESM-2",
                            "model": model_name,
                            "mode": mode,
                            "ambiguity_ratio": ratio,
                            "fold": fold_idx,
                            "y_true": y_test,
                            "y_score": y_score,
                        }
                    )
                    print(
                        "Ambiguity | {} | {} | ratio {:.2f} | fold {}: AUC={:.3f}".format(
                            mode,
                            model_name,
                            ratio,
                            fold_idx,
                            metrics["roc_auc"],
                        )
                    )
    return pd.DataFrame(rows), roc_records


def summarize_ambiguity_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "f1", "mcc", "roc_auc", "average_precision"]
    rows = []
    for (model, mode, ratio), group in metrics.groupby(["model", "mode", "ambiguity_ratio"], sort=False):
        row = {"feature": "ESM-2", "model": model, "mode": mode, "ambiguity_ratio": ratio}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = 1.96 * group[metric].std(ddof=1) / np.sqrt(len(group))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ambiguity_roc(roc_records: list[dict[str, object]], outdir: Path, ratios: list[float]) -> None:
    mean_fpr = np.linspace(0, 1, 201)
    ratio_colors = {
        ratio: plt.get_cmap("coolwarm")(value)
        for ratio, value in zip(ratios, np.linspace(0.08, 0.92, len(ratios)))
    }

    for model_name in AMBIGUITY_MODELS:
        fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.2), sharex=True, sharey=True)
        for ax, mode in zip(axes, ["negative-noise", "positive-noise"]):
            for ratio in ratios:
                records = [
                    record
                    for record in roc_records
                    if record["model"] == model_name
                    and record["mode"] == mode
                    and np.isclose(record["ambiguity_ratio"], ratio)
                ]
                if not records:
                    continue
                roc_stats = mean_roc_with_ci(records, mean_fpr)
                mean_tpr = roc_stats["mean_tpr"]
                tpr_ci = roc_stats["tpr_ci"]
                mean_auc = roc_stats["mean_auc"]
                auc_ci = roc_stats["auc_ci"]
                color = ratio_colors[ratio]
                ax.plot(
                    mean_fpr,
                    mean_tpr,
                    color=color,
                    linewidth=1.6,
                    label=f"{ratio:.0%} AUC={mean_auc:.3f}±{auc_ci:.3f}",
                )
                ax.fill_between(
                    mean_fpr,
                    np.maximum(mean_tpr - tpr_ci, 0),
                    np.minimum(mean_tpr + tpr_ci, 1),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )
            ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.0)
            ax.set_title(mode)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.grid(color="#E6E6E6", linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=True, edgecolor="black", fontsize=6.6, loc="lower right")

        fig.suptitle(f"ESM-2 + {model_name} ambiguity perturbation", y=1.02)
        fig.tight_layout()
        fig.savefig(
            outdir / f"05_ambiguity_roc_esm2_{model_name.lower()}.png",
            dpi=400,
            transparent=True,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_ambiguity_pipeline(outdir: Path) -> None:
    text = """# Ambiguity perturbation pipeline

本脚本已将 -1 ambiguity 作为模型鲁棒性扰动，而不是直接混入主 baseline。

1. 固定同一个 0/1 平衡训练、验证、测试划分，保持 test set 只包含高置信 0/1 标签。
2. 设定 ambiguity 注入比例，例如 0%、10%、25%、50%、100%，比例以训练集中阳性样本数为基准。
3. 每个比例下，从 -1 样本中随机抽取对应数量并加入训练集，不加入验证集和测试集。
4. 为避免人为指定 -1 的真实标签，可设计两条扰动路径：
   - negative-noise：将 -1 临时视作 0，测试模型是否被弱 DBP 信号污染。
   - positive-noise：将 -1 临时视作 1，测试模型是否对弱 DBP 信号过敏。
5. 当前执行模型为 ESM-2 + LR 和 ESM-2 + SVM；每个比例和每条路径在 9 折训练中重复评估，记录 ROC-AUC、F1、MCC 和 average precision 的变化。
6. 以 0% ambiguity 的 baseline 为参照，绘制指标变化曲线；若 ESM-2 的性能下降更小，说明其表示对模糊注释扰动更稳健。
"""
    (outdir / "04_ambiguity_perturbation_pipeline.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    ambiguity_ratios = parse_ratios(args.ambiguity_ratios)
    prepare_output_dir(outdir)

    df = pd.read_csv(input_path, sep="\t")
    required = {"protein_id", "is_DBP", "embedded_sequence", "average_pooling_embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Missing required columns: {}".format(", ".join(sorted(missing))))

    balanced = make_balanced_binary_dataset(df, args.random_state)
    train_valid, test = train_test_split(
        balanced,
        test_size=args.test_size,
        stratify=balanced["is_DBP"],
        random_state=args.random_state,
    )
    split_summary = pd.DataFrame(
        [
            {"split": "all_labeled", "label": label, "count": int((df["is_DBP"] == label).sum())}
            for label in [-1, 0, 1]
        ]
        + [
            {"split": "balanced_binary", "label": label, "count": int((balanced["is_DBP"] == label).sum())}
            for label in [0, 1]
        ]
        + [
            {"split": "train_valid", "label": label, "count": int((train_valid["is_DBP"] == label).sum())}
            for label in [0, 1]
        ]
        + [
            {"split": "test", "label": label, "count": int((test["is_DBP"] == label).sum())}
            for label in [0, 1]
        ]
    )
    split_summary.to_csv(outdir / "00_dataset_split_summary.csv", index=False)

    fold_metrics = []
    roc_records = []
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    y_train_valid = train_valid["is_DBP"].astype(int).to_numpy()
    y_test = test["is_DBP"].astype(int).to_numpy()

    for feature_name, feature_col in FEATURE_COLUMNS.items():
        x_train_valid = vector_matrix(train_valid[feature_col])
        x_test = vector_matrix(test[feature_col])
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train_valid, y_train_valid), start=1):
            models = make_models(args.random_state + fold_idx, args.rf_trees, args.mlp_max_iter)
            x_train = x_train_valid[train_idx]
            y_train = y_train_valid[train_idx]
            valid_counts = np.bincount(y_train_valid[valid_idx], minlength=2)
            train_counts = np.bincount(y_train, minlength=2)

            for model_name in MODEL_ORDER:
                model = models[model_name]
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                y_score = model_scores(model, x_test)
                metrics = evaluate_predictions(y_test, y_pred, y_score)
                fold_metrics.append(
                    {
                        "feature": feature_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "train_n0": int(train_counts[0]),
                        "train_n1": int(train_counts[1]),
                        "valid_n0": int(valid_counts[0]),
                        "valid_n1": int(valid_counts[1]),
                        "test_n0": int((y_test == 0).sum()),
                        "test_n1": int((y_test == 1).sum()),
                        **metrics,
                    }
                )
                roc_records.append(
                    {
                        "feature": feature_name,
                        "model": model_name,
                        "fold": fold_idx,
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "y_score": y_score,
                    }
                )
                print(f"{feature_name} | {model_name} | fold {fold_idx}: AUC={metrics['roc_auc']:.3f}")

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(outdir / "01_test_metrics_by_fold.csv", index=False)
    summary = summarize_metrics(fold_metrics_df)
    summary.to_csv(outdir / "01_test_metrics_summary_mean_sd.csv", index=False)
    plot_roc_curves(roc_records, outdir)
    plot_roc_pr_panel(roc_records, outdir)
    plot_confusion_matrices(roc_records, outdir)

    ambiguity_metrics, ambiguity_roc_records = run_ambiguity_test(
        df=df,
        train_valid=train_valid,
        test=test,
        cv=cv,
        ratios=ambiguity_ratios,
        random_state=args.random_state,
    )
    ambiguity_metrics.to_csv(outdir / "04_ambiguity_metrics_by_fold.csv", index=False)
    ambiguity_summary = summarize_ambiguity_metrics(ambiguity_metrics)
    ambiguity_summary.to_csv(outdir / "04_ambiguity_metrics_summary_ci95.csv", index=False)
    plot_ambiguity_roc(ambiguity_roc_records, outdir, ambiguity_ratios)
    save_ambiguity_pipeline(outdir)

    print(f"Split summary: {outdir / '00_dataset_split_summary.csv'}")
    print(f"Fold metrics: {outdir / '01_test_metrics_by_fold.csv'}")
    print(f"Metric summary: {outdir / '01_test_metrics_summary_mean_sd.csv'}")
    print(f"ROC figure: {outdir / '02_test_roc_auc_curves.png'}")
    print(f"ROC/PR panel: {outdir / '06_roc_pr_auc_panel.png'}")
    print(f"Confusion matrix panel: {outdir / '07_confusion_matrix_panel.png'}")
    print(f"Ambiguity fold metrics: {outdir / '04_ambiguity_metrics_by_fold.csv'}")
    print(f"Ambiguity metric summary: {outdir / '04_ambiguity_metrics_summary_ci95.csv'}")
    print(f"Ambiguity LR ROC: {outdir / '05_ambiguity_roc_esm2_lr.png'}")
    print(f"Ambiguity SVM ROC: {outdir / '05_ambiguity_roc_esm2_svm.png'}")
    print(f"Ambiguity pipeline: {outdir / '04_ambiguity_perturbation_pipeline.md'}")


if __name__ == "__main__":
    main()
