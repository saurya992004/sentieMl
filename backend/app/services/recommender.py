from typing import Any
import pandas as pd
from .profiler import profile_dataset


def recommend_models(
    frame: pd.DataFrame,
    target_column: str,
    problem_type: str | None = None,
) -> dict[str, Any]:
    row_count, col_count = frame.shape
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' does not exist in dataset")

    # Determine or validate problem type
    target_s = frame[target_column].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(target_s)
    unique_targets = int(target_s.nunique())

    if not problem_type:
        if not is_numeric or unique_targets <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"

    # Profile summary
    profile = profile_dataset(frame, target_column=target_column)
    num_cols = len(profile["numerical_columns"])
    cat_cols = len(profile["categorical_columns"])
    duplicate_rows = profile["duplicate_rows"]
    target_info = profile["target_analysis"] or {}
    is_imbalanced = target_info.get("is_imbalanced", False)
    has_high_outliers = any(o["outlier_percentage"] > 5.0 for o in profile["outliers"].values())

    recommendations = []

    if problem_type == "classification":
        # 1. Logistic Regression
        lr_score = 75.0
        lr_reasons = []
        if row_count < 500:
            lr_score += 10.0
            lr_reasons.append("Small sample size: linear models resist overfitting on small datasets.")
        if num_cols > cat_cols:
            lr_score += 5.0
            lr_reasons.append("Mostly numerical features: well-suited for linear decision boundaries.")
        if is_imbalanced:
            lr_score -= 10.0
            lr_reasons.append("Severe class imbalance: may require careful threshold tuning.")
        if has_high_outliers:
            lr_score -= 5.0
            lr_reasons.append("Outliers detected: linear models can be sensitive to extreme feature values.")
        lr_reasons.append("Fastest training and lowest inference latency; excellent interpretable baseline.")

        recommendations.append({
            "algorithm": "logistic_regression",
            "display_name": "Logistic Regression",
            "score": round(max(30.0, min(95.0, lr_score)), 1),
            "recommended": lr_score >= 70.0,
            "reasons": lr_reasons,
            "estimated_speed": "Very Fast (< 1s)",
            "computational_requirement": "Low",
        })

        # 2. Random Forest Classifier
        rf_score = 85.0
        rf_reasons = []
        if cat_cols > 0:
            rf_score += 5.0
            rf_reasons.append("Handles categorical and mixed feature distributions naturally.")
        if has_high_outliers:
            rf_score += 5.0
            rf_reasons.append("Decision trees are invariant to monotonic transformations and robust to outliers.")
        if is_imbalanced:
            rf_score += 5.0
            rf_reasons.append("Ensemble bagging handles non-uniform class distributions effectively.")
        if row_count > 50000:
            rf_score -= 5.0
            rf_reasons.append("Large dataset: multi-tree bagging can increase memory consumption.")
        rf_reasons.append("High general accuracy with minimal hyperparameter tuning required.")

        recommendations.append({
            "algorithm": "random_forest",
            "display_name": "Random Forest Classifier",
            "score": round(max(40.0, min(98.0, rf_score)), 1),
            "recommended": rf_score >= 75.0,
            "reasons": rf_reasons,
            "estimated_speed": "Moderate (1 - 5s)",
            "computational_requirement": "Medium",
        })

        # 3. Gradient Boosting Classifier
        gb_score = 88.0
        gb_reasons = []
        if row_count >= 500:
            gb_score += 5.0
            gb_reasons.append("Adequate sample size: gradient boosting excels at learning complex non-linear patterns.")
        else:
            gb_score -= 10.0
            gb_reasons.append("Small sample size: sequential boosting can risk overfitting if unconstrained.")
        if is_imbalanced:
            gb_score += 5.0
            gb_reasons.append("Gradient boosting iteratively corrects hard-to-classify minority samples.")
        gb_reasons.append("Often produces top predictive performance on tabular benchmarks.")

        recommendations.append({
            "algorithm": "gradient_boosting",
            "display_name": "Gradient Boosting Classifier",
            "score": round(max(40.0, min(99.0, gb_score)), 1),
            "recommended": gb_score >= 75.0,
            "reasons": gb_reasons,
            "estimated_speed": "Moderate to Slow (3 - 15s)",
            "computational_requirement": "Medium to High",
        })

    else:
        # Regression
        # 1. Linear Regression
        lin_score = 70.0
        lin_reasons = []
        if row_count < 1000:
            lin_score += 10.0
            lin_reasons.append("Fast, stable, and transparent baseline for continuous targets.")
        if has_high_outliers:
            lin_score -= 10.0
            lin_reasons.append("Sensitive to target and feature outliers via squared error penalty.")
        lin_reasons.append("Near-instant inference time; ideal for low-latency operational deployment.")

        recommendations.append({
            "algorithm": "linear_regression",
            "display_name": "Linear Regression",
            "score": round(max(30.0, min(90.0, lin_score)), 1),
            "recommended": lin_score >= 65.0,
            "reasons": lin_reasons,
            "estimated_speed": "Very Fast (< 1s)",
            "computational_requirement": "Low",
        })

        # 2. Random Forest Regressor
        rf_score = 85.0
        rf_reasons = []
        if has_high_outliers:
            rf_score += 5.0
            rf_reasons.append("Tree splits bound predictions naturally, mitigating outlier shock.")
        if num_cols + cat_cols > 5:
            rf_score += 5.0
            rf_reasons.append("Captures multi-feature non-linear interactions without polynomial expansion.")
        rf_reasons.append("Balanced trade-off between variance reduction and non-linear fitting.")

        recommendations.append({
            "algorithm": "random_forest",
            "display_name": "Random Forest Regressor",
            "score": round(max(40.0, min(98.0, rf_score)), 1),
            "recommended": rf_score >= 75.0,
            "reasons": rf_reasons,
            "estimated_speed": "Moderate (1 - 5s)",
            "computational_requirement": "Medium",
        })

        # 3. Gradient Boosting Regressor
        gb_score = 90.0
        gb_reasons = []
        if row_count >= 500:
            gb_score += 5.0
            gb_reasons.append("Sequential residual fitting typically achieves the lowest RMSE and MAE.")
        else:
            gb_score -= 10.0
            gb_reasons.append("May overfit small datasets without strict early stopping.")
        gb_reasons.append("State-of-the-art tabular regression performance.")

        recommendations.append({
            "algorithm": "gradient_boosting",
            "display_name": "Gradient Boosting Regressor",
            "score": round(max(40.0, min(99.0, gb_score)), 1),
            "recommended": gb_score >= 75.0,
            "reasons": gb_reasons,
            "estimated_speed": "Moderate to Slow (3 - 15s)",
            "computational_requirement": "Medium to High",
        })

    # Sort recommendations by score descending
    recommendations.sort(key=lambda x: x["score"], reverse=True)

    summary_text = (
        f"For dataset with {row_count} rows and {col_count} columns targeting '{target_column}' ({problem_type}): "
        f"Top recommendation is {recommendations[0]['display_name']} (score {recommendations[0]['score']}/100). "
        f"{'Class imbalance was detected; non-linear ensemble models are favored.' if is_imbalanced else 'Balanced tabular distribution.'}"
    )

    return {
        "dataset_id": -1,  # caller will fill this
        "target_column": target_column,
        "problem_type": problem_type,
        "recommended_algorithms": recommendations,
        "explanation": summary_text,
        "dataset_summary": {
            "row_count": row_count,
            "column_count": col_count,
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "is_imbalanced": is_imbalanced,
            "has_high_outliers": has_high_outliers,
        },
    }
