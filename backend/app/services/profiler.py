from typing import Any
import numpy as np
import pandas as pd


def _clean_val(v: Any) -> Any:
    if isinstance(v, (np.integer, np.floating)):
        if np.isnan(v) or np.isinf(v):
            return None
        return v.item()
    if isinstance(v, np.ndarray):
        return [_clean_val(x) for x in v.tolist()]
    if pd.isna(v):
        return None
    return v


def profile_dataset(frame: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    row_count, col_count = frame.shape
    num_cols = frame.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in frame.columns if c not in num_cols]
    
    missing_counts = {str(c): int(frame[c].isna().sum()) for c in frame.columns}
    missing_pcts = {str(c): float(round(missing_counts[str(c)] / max(1, row_count) * 100, 2)) for c in frame.columns}
    duplicate_rows = int(frame.duplicated().sum())
    unique_counts = {str(c): int(frame[c].nunique(dropna=True)) for c in frame.columns}

    # Descriptive statistics
    stats: dict[str, dict[str, Any]] = {}
    for c in num_cols:
        series = frame[c].dropna()
        if len(series) > 0:
            stats[c] = {
                "type": "numeric",
                "count": int(series.count()),
                "mean": _clean_val(round(float(series.mean()), 4)),
                "std": _clean_val(round(float(series.std()), 4)) if len(series) > 1 else 0.0,
                "min": _clean_val(round(float(series.min()), 4)),
                "p25": _clean_val(round(float(series.quantile(0.25)), 4)),
                "median": _clean_val(round(float(series.median()), 4)),
                "p75": _clean_val(round(float(series.quantile(0.75)), 4)),
                "max": _clean_val(round(float(series.max()), 4)),
                "skew": _clean_val(round(float(series.skew()), 4)) if len(series) > 2 else 0.0,
            }
        else:
            stats[c] = {"type": "numeric", "count": 0, "mean": None, "std": None, "min": None, "p25": None, "median": None, "p75": None, "max": None, "skew": None}

    for c in cat_cols:
        series = frame[c].dropna()
        top_val = str(series.mode().iloc[0]) if not series.empty else None
        top_freq = int(series.value_counts().iloc[0]) if not series.empty else 0
        stats[c] = {
            "type": "categorical",
            "count": int(series.count()),
            "unique": unique_counts[c],
            "top": top_val,
            "top_freq": top_freq,
            "top_pct": round(top_freq / max(1, row_count) * 100, 2) if row_count > 0 else 0.0,
            "categories": [str(x) for x in series.unique()[:20]],
        }

    # Outlier Analysis using IQR
    outliers: dict[str, dict[str, Any]] = {}
    for c in num_cols:
        series = frame[c].dropna()
        if len(series) >= 4:
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            lower_outliers = int((series < lower_bound).sum())
            upper_outliers = int((series > upper_bound).sum())
            total_outliers = lower_outliers + upper_outliers
            outliers[c] = {
                "lower_bound": _clean_val(round(lower_bound, 4)),
                "upper_bound": _clean_val(round(upper_bound, 4)),
                "outlier_count": total_outliers,
                "outlier_percentage": round(total_outliers / max(1, len(series)) * 100, 2),
            }
        else:
            outliers[c] = {"lower_bound": None, "upper_bound": None, "outlier_count": 0, "outlier_percentage": 0.0}

    # Correlation Analysis
    correlations: dict[str, dict[str, float]] = {}
    if len(num_cols) > 1:
        corr_matrix = frame[num_cols].corr()
        for col1 in num_cols:
            correlations[col1] = {}
            for col2 in num_cols:
                val = corr_matrix.loc[col1, col2]
                correlations[col1][col2] = _clean_val(round(float(val), 4)) if pd.notna(val) else 0.0

    # Target Variable Analysis
    target_analysis: dict[str, Any] | None = None
    if target_column and target_column in frame.columns:
        target_s = frame[target_column].dropna()
        is_num = target_column in num_cols
        uniques = int(target_s.nunique())
        dist = {str(k): int(v) for k, v in target_s.value_counts().head(20).items()}
        imbalance_ratio = 1.0
        if not is_num or uniques <= 10:
            counts = list(dist.values())
            if len(counts) > 1 and min(counts) > 0:
                imbalance_ratio = round(max(counts) / min(counts), 2)
        target_analysis = {
            "target_column": target_column,
            "type": "numeric" if is_num and uniques > 10 else "categorical",
            "unique_count": uniques,
            "distribution": dist,
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio > 3.0,
        }

    # Dataset Quality Warnings & Rules
    warnings: list[dict[str, Any]] = []
    suggested_operations: list[str] = []
    risks: list[str] = []
    score_deductions = 0.0

    # 1. Missing values
    for col, pct in missing_pcts.items():
        if pct > 50.0:
            warnings.append({"column": col, "severity": "HIGH", "message": f"Feature '{col}' has {pct}% missing values."})
            score_deductions += 15.0
            suggested_operations.append(f"Consider dropping feature '{col}' due to severe missingness (>50%).")
        elif pct > 15.0:
            warnings.append({"column": col, "severity": "MEDIUM", "message": f"Feature '{col}' has {pct}% missing values."})
            score_deductions += 8.0
            suggested_operations.append(f"Impute missing values in '{col}' using {'median' if col in num_cols else 'mode'}.")
        elif pct > 0.0:
            suggested_operations.append(f"Impute missing values in '{col}'.")

    # 2. Duplicate rows
    if duplicate_rows > 0:
        dup_pct = round(duplicate_rows / max(1, row_count) * 100, 2)
        if dup_pct > 5.0:
            warnings.append({"column": "All", "severity": "HIGH", "message": f"Dataset contains {duplicate_rows} duplicate rows ({dup_pct}%)."})
            score_deductions += 12.0
            risks.append(f"High duplicate rate ({dup_pct}%) may cause data leakage between train and test sets.")
            suggested_operations.append("Deduplicate dataset prior to training.")
        else:
            warnings.append({"column": "All", "severity": "LOW", "message": f"Dataset contains {duplicate_rows} duplicate rows."})
            score_deductions += 3.0
            suggested_operations.append("Review and remove duplicate rows.")

    # 3. Constant columns & ID columns
    for col in frame.columns:
        if unique_counts[col] <= 1:
            warnings.append({"column": col, "severity": "HIGH", "message": f"Feature '{col}' is constant (zero variance)."})
            score_deductions += 8.0
            suggested_operations.append(f"Drop constant column '{col}'.")
            risks.append(f"Constant column '{col}' provides no predictive signal.")
        elif unique_counts[col] == row_count and col in cat_cols:
            warnings.append({"column": col, "severity": "MEDIUM", "message": f"Feature '{col}' has 100% unique values (likely an ID or key)."})
            score_deductions += 5.0
            suggested_operations.append(f"Drop identifier column '{col}'.")
            risks.append(f"Identifier column '{col}' may cause memorization or overfitting.")

    # 4. Multicollinearity / Highly correlated features
    high_corr_pairs = []
    for c1 in correlations:
        for c2 in correlations[c1]:
            if c1 < c2:
                val = abs(correlations[c1][c2])
                if val >= 0.85:
                    high_corr_pairs.append((c1, c2, val))
                    warnings.append({"column": f"{c1} & {c2}", "severity": "MEDIUM", "message": f"High collinearity between '{c1}' and '{c2}' (r={val})."})
                    score_deductions += 4.0
    if high_corr_pairs:
        suggested_operations.append("Consider feature selection or dimensionality reduction for highly correlated pairs.")
        risks.append("Collinear features can destabilize linear model coefficients and inflate feature variance.")

    # 5. Outliers
    high_outlier_cols = [c for c, o in outliers.items() if o["outlier_percentage"] > 5.0]
    if high_outlier_cols:
        warnings.append({"column": ", ".join(high_outlier_cols), "severity": "LOW", "message": f"Features {high_outlier_cols} have >5% outliers."})
        suggested_operations.append("Apply robust scaling or tree-based algorithms less sensitive to extreme outliers.")

    # 6. Target Analysis & Leakage Check
    if target_analysis:
        if target_analysis.get("is_imbalanced"):
            warnings.append({"column": target_column, "severity": "HIGH", "message": f"Target '{target_column}' is imbalanced (ratio {target_analysis['imbalance_ratio']}:1)."})
            score_deductions += 10.0
            risks.append("Class imbalance will bias models towards the majority class; use F1 or ROC-AUC rather than raw accuracy.")
            suggested_operations.append("Use stratified splits and class-weighted loss or resampling (e.g. SMOTE).")
        
        # Data leakage check: near-perfect correlation with target
        if target_column in num_cols:
            for c in num_cols:
                if c != target_column and abs(correlations.get(c, {}).get(target_column, 0.0)) > 0.98:
                    warnings.append({"column": c, "severity": "CRITICAL", "message": f"Potential target leakage: '{c}' has r={correlations[c][target_column]} with target '{target_column}'."})
                    score_deductions += 25.0
                    risks.append(f"Feature '{c}' is almost perfectly correlated with the target; verify it is available at inference time.")

    # Compute Quality Score (0 to 100)
    quality_score = max(0.0, min(100.0, round(100.0 - score_deductions, 1)))

    if not suggested_operations:
        suggested_operations.append("Dataset is clean. Standard scaling and encoding are sufficient.")

    # Generate dynamic LLM insights via Gemini (with graceful fallback)
    ai_summary = None
    try:
        from .llm import get_ai_data_profile_summary
        ai_summary = get_ai_data_profile_summary(
            quality_score=quality_score,
            row_count=row_count,
            col_count=col_count,
            warnings=warnings,
            statistics=stats,
            outliers=outliers,
            target_info=target_analysis,
        )
    except Exception as exc:
        logger.debug("AI profiling summary fallback: %s", exc)

    return {
        "row_count": row_count,
        "column_count": col_count,
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_counts": missing_counts,
        "missing_percentages": missing_pcts,
        "duplicate_rows": duplicate_rows,
        "unique_counts": unique_counts,
        "statistics": stats,
        "outliers": outliers,
        "correlations": correlations,
        "target_analysis": target_analysis,
        "quality_score": quality_score,
        "warnings": warnings,
        "suggested_operations": suggested_operations,
        "risks": risks,
        "ai_summary": ai_summary,
    }
