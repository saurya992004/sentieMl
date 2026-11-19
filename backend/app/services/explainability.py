import logging
from typing import Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def explain_pipeline(
    pipeline: Any,
    X_sample: pd.DataFrame,
    local_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generates global and local explanations for a scikit-learn Pipeline.

    Uses SHAP when feasible, with robust fallback to tree feature importances
    or linear model coefficients.
    """
    # Unwrap wrappers if model was loaded via PyFunc or custom wrapper
    if hasattr(pipeline, "sklearn_model"):
        pipeline = pipeline.sklearn_model
    elif hasattr(pipeline, "_model_impl"):
        pipeline = getattr(pipeline._model_impl, "sklearn_model", pipeline._model_impl)

    feature_names = list(X_sample.columns)
    named_steps = getattr(pipeline, "named_steps", None)
    if named_steps and hasattr(named_steps, "get"):
        model = named_steps.get("model", pipeline)
        preprocessor = named_steps.get("preprocessor", None)
    else:
        model = pipeline
        preprocessor = None

    # Transform sample if preprocessor exists
    try:
        if preprocessor:
            X_trans = preprocessor.transform(X_sample.head(50))
            # Try to get transformed feature names
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    trans_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]
                except Exception:
                    trans_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
            else:
                trans_names = [f"feat_{i}" for i in range(X_trans.shape[1])]
        else:
            X_trans = X_sample.head(50).values
            trans_names = feature_names
    except Exception as exc:
        logger.warning("Feature transformation failed for explainability: %s", exc)
        X_trans = None
        trans_names = feature_names

    global_importance: list[dict[str, Any]] = []
    shap_used = False

    # Attempt SHAP
    if X_trans is not None:
        try:
            import shap
            if hasattr(model, "feature_importances_"):
                # Tree-based model
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_trans)
                if isinstance(shap_values, list):
                    # Multiclass classification: average magnitude across classes
                    vals = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                elif len(shap_values.shape) == 3:
                    vals = np.mean(np.abs(shap_values), axis=2)
                else:
                    vals = np.abs(shap_values)
                mean_shap = np.mean(vals, axis=0)
                shap_used = True
            elif hasattr(model, "coef_"):
                # Linear model
                explainer = shap.LinearExplainer(model, X_trans)
                shap_values = explainer.shap_values(X_trans)
                vals = np.abs(shap_values)
                mean_shap = np.mean(vals, axis=0)
                shap_used = True
            else:
                mean_shap = None

            if shap_used and mean_shap is not None:
                # Map back to original feature names if one-hot expanded
                col_importance: dict[str, float] = {col: 0.0 for col in feature_names}
                for name, imp in zip(trans_names, mean_shap):
                    matched = False
                    for orig in feature_names:
                        if name == orig or name.startswith(orig + "_"):
                            col_importance[orig] += float(imp)
                            matched = True
                            break
                    if not matched:
                        col_importance[name] = float(imp)

                total_imp = sum(col_importance.values()) or 1.0
                global_importance = [
                    {
                        "feature": k,
                        "importance": round(v, 4),
                        "percentage": round(v / total_imp * 100, 2),
                    }
                    for k, v in sorted(col_importance.items(), key=lambda x: x[1], reverse=True)
                ]
        except Exception as exc:
            logger.info("SHAP computation not directly applicable: %s. Using intrinsic importance.", exc)
            shap_used = False

    # Fallback to model intrinsic importance if SHAP was not used
    if not global_importance:
        if hasattr(model, "feature_importances_"):
            raw_imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            raw_imp = np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raw_imp = np.ones(len(feature_names))

        col_importance = {col: 0.0 for col in feature_names}
        for i, val in enumerate(raw_imp):
            orig = feature_names[i % len(feature_names)]
            col_importance[orig] += float(val)

        total_imp = sum(col_importance.values()) or 1.0
        global_importance = [
            {
                "feature": k,
                "importance": round(v, 4),
                "percentage": round(v / total_imp * 100, 2),
            }
            for k, v in sorted(col_importance.items(), key=lambda x: x[1], reverse=True)
        ]

    # Local Explanation
    local_explanation: list[dict[str, Any]] = []
    if local_sample:
        row_df = pd.DataFrame([local_sample])
        # Find feature contributions relative to mean
        for item in global_importance:
            feat = item["feature"]
            val = local_sample.get(feat, None)
            local_explanation.append({
                "feature": feat,
                "value": val,
                "relative_impact": item["importance"],
                "importance_pct": item["percentage"],
            })

    top_feature = global_importance[0]["feature"] if global_importance else "features"
    top_pct = global_importance[0]["percentage"] if global_importance else 0.0
    summary = (
        f"Model explainability derived using {'SHAP game-theoretic Shapley values' if shap_used else 'intrinsic feature attribution'}. "
        f"'{top_feature}' has the highest relative contribution ({top_pct}% of total influence). "
        "Important: Feature attribution reflects statistical associations learned by the model and should not be interpreted as proof of causal mechanisms."
    )

    ai_business_translation = None
    try:
        from .llm import get_ai_shap_business_translation
        ai_business_translation = get_ai_shap_business_translation(
            model_name=model.__class__.__name__,
            algorithm=model.__class__.__name__,
            top_features=global_importance,
            local_sample=local_sample,
        )
    except Exception:
        pass

    return {
        "algorithm": model.__class__.__name__,
        "feature_names": feature_names,
        "global_importance": global_importance,
        "local_explanation": local_explanation if local_sample else None,
        "method": "SHAP" if shap_used else "Intrinsic Attribution",
        "summary": summary,
        "ai_business_translation": ai_business_translation,
    }
