from datetime import datetime, timezone
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from ..models import DriftReport, ModelLifecycle, ProductionPrediction


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between reference and production."""
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    try:
        # Quantile buckets based on expected
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Unique bins
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0

        exp_counts, _ = np.histogram(expected, bins=bins)
        act_counts, _ = np.histogram(actual, bins=bins)

        exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts) / len(expected)
        act_pct = np.where(act_counts == 0, 0.0001, act_counts) / len(actual)

        psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
        return float(round(psi, 4))
    except Exception:
        return 0.0


def analyze_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> dict[str, Any]:
    """Analyzes numerical and categorical distribution drift between reference and production data."""
    common_cols = [c for c in reference_df.columns if c in production_df.columns]
    feature_drift: dict[str, Any] = {}
    drifted_count = 0

    num_cols = reference_df[common_cols].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in common_cols if c not in num_cols]

    # 1. Numerical drift via 2-sample Kolmogorov-Smirnov test and PSI
    for col in num_cols:
        ref_s = reference_df[col].dropna().values
        prod_s = production_df[col].dropna().values
        if len(ref_s) < 2 or len(prod_s) < 2:
            continue
        try:
            ks_res = stats.ks_2samp(ref_s, prod_s)
            p_val = float(ks_res.pvalue)
            stat = float(ks_res.statistic)
            psi = calculate_psi(ref_s, prod_s)
            is_drifted = p_val < p_value_threshold or psi > 0.2
            if is_drifted:
                drifted_count += 1
            feature_drift[col] = {
                "type": "numeric",
                "test": "Kolmogorov-Smirnov + PSI",
                "statistic": round(stat, 4),
                "p_value": round(p_val, 4),
                "psi": psi,
                "drift_detected": is_drifted,
                "ref_mean": round(float(np.mean(ref_s)), 4),
                "prod_mean": round(float(np.mean(prod_s)), 4),
            }
        except Exception:
            pass

    # 2. Categorical drift via Chi-square or distribution divergence
    for col in cat_cols:
        ref_vc = reference_df[col].dropna().value_counts(normalize=True)
        prod_vc = production_df[col].dropna().value_counts(normalize=True)
        all_cats = list(set(ref_vc.index).union(set(prod_vc.index)))
        if not all_cats:
            continue
        p = np.array([ref_vc.get(c, 1e-4) for c in all_cats])
        q = np.array([prod_vc.get(c, 1e-4) for c in all_cats])
        p /= p.sum()
        q /= q.sum()
        # Total Variation Distance
        tvd = float(np.sum(np.abs(p - q)) / 2.0)
        is_drifted = tvd > 0.15
        if is_drifted:
            drifted_count += 1
        feature_drift[col] = {
            "type": "categorical",
            "test": "Total Variation Distance",
            "statistic": round(tvd, 4),
            "p_value": None,
            "psi": round(tvd, 4),
            "drift_detected": is_drifted,
            "categories": len(all_cats),
        }

    total_checked = len(feature_drift)
    drift_ratio = (drifted_count / total_checked) if total_checked > 0 else 0.0

    if drift_ratio >= 0.5:
        severity = "CRITICAL"
        action = "High dataset drift detected. Immediate model retraining recommended."
    elif drift_ratio >= 0.25:
        severity = "MODERATE"
        action = "Moderate drift detected across several features. Monitor closely and plan retraining."
    elif drifted_count > 0:
        severity = "LOW"
        action = "Minor drift detected on isolated features. Model performance remains acceptable."
    else:
        severity = "NONE"
        action = "No significant feature drift detected. Production data matches training baseline."

    return {
        "sample_count": len(production_df),
        "total_features": total_checked,
        "drifted_features_count": drifted_count,
        "overall_drift_score": round(drift_ratio, 4),
        "drift_detected": drifted_count > 0,
        "drift_severity": severity,
        "recommended_action": action,
        "feature_drift": feature_drift,
    }


def evaluate_degradation(
    baseline_metrics: dict[str, Any],
    production_predictions: list[dict[str, Any]],
    problem_type: str = "classification",
) -> dict[str, Any]:
    """Calculates production performance when ground truth labels are provided and compares with baseline."""
    labeled = [p for p in production_predictions if p.get("ground_truth") is not None]
    if not labeled:
        return {
            "status": "NO_GROUND_TRUTH",
            "message": "Ground-truth labels are not yet available for production predictions.",
            "sample_count": 0,
        }

    y_true = [p["ground_truth"] for p in labeled]
    y_pred = [p["prediction"] for p in labeled]

    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

    current_metrics: dict[str, float] = {}
    if problem_type == "classification":
        current_metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        current_metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    else:
        current_metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        current_metrics["rmse"] = float(mean_squared_error(y_true, y_pred) ** 0.5)
        current_metrics["r2"] = float(r2_score(y_true, y_pred))

    degradation_detected = False
    details = []

    # Check for degradation against baseline
    for metric_name, curr_val in current_metrics.items():
        base_val = baseline_metrics.get(metric_name)
        if base_val is not None:
            base_val = float(base_val)
            if metric_name in {"mae", "rmse"}:
                # Increase in error indicates degradation
                diff_pct = (curr_val - base_val) / max(0.001, base_val) * 100.0
                if diff_pct > 15.0:
                    degradation_detected = True
                    details.append(f"{metric_name.upper()} increased by {round(diff_pct, 1)}% (baseline: {round(base_val, 4)}, current: {round(curr_val, 4)})")
            else:
                # Decrease in accuracy/F1/R2 indicates degradation
                diff_pct = (base_val - curr_val) / max(0.001, base_val) * 100.0
                if diff_pct > 10.0:
                    degradation_detected = True
                    details.append(f"{metric_name.upper()} dropped by {round(diff_pct, 1)}% (baseline: {round(base_val, 4)}, current: {round(curr_val, 4)})")

    return {
        "sample_count": len(labeled),
        "current_metrics": current_metrics,
        "baseline_metrics": baseline_metrics,
        "degradation_detected": degradation_detected,
        "status": "DEGRADATION_DETECTED" if degradation_detected else "HEALTHY",
        "degradation_details": details,
        "recommended_action": "Model performance has dropped below threshold. Retraining workflow should be triggered." if degradation_detected else "Model performance aligns with baseline expectations.",
    }
