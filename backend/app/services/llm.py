import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any
from ..config import settings

logger = logging.getLogger(__name__)


def get_effective_gemini_key(user_key: str | None = None) -> str | None:
    if user_key and user_key.strip():
        return user_key.strip()
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        return settings.gemini_api_key.strip()
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def call_gemini(prompt: str, api_key: str | None = None, timeout: float = 10.0) -> str | None:
    key = get_effective_gemini_key(api_key)
    if not key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000,
        }
    }

    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as exc:
        logger.warning("Gemini API call failed (%s). Falling back to statistical heuristics.", exc)
        return None
    return None


def get_ai_data_profile_summary(
    quality_score: float,
    row_count: int,
    col_count: int,
    warnings: list[dict[str, Any]],
    statistics: dict[str, Any],
    outliers: dict[str, Any],
    target_info: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> str | None:
    """Generates an executive-level dynamic data intelligence narrative via Gemini."""
    prompt = f"""You are an expert Chief Data Scientist evaluating a production dataset for machine learning.
Analyze the following empirical dataset statistics:
- Total Observations: {row_count} rows, {col_count} columns
- Overall Data Quality Score: {quality_score}/100
- Warnings Detected: {json.dumps(warnings[:5])}
- Total Outlier Columns: {len(outliers.get('counts', {}))}
- Target Column Analysis: {json.dumps(target_info) if target_info else 'Unsupervised / Not specified'}

Please provide a concise, high-impact executive data intelligence brief (3-4 concise paragraphs):
1. Executive Quality & Hygiene Assessment: Synthesize overall data health and reliability.
2. Critical Data Bottlenecks & Leakage Risks: Highlight severe missingness, extreme outliers, or class imbalance.
3. Feature Engineering & Preprocessing Strategy: Specific transformations (e.g. log scaling, frequency encoding, robust clipping).
Write in an authoritative, clear, and professional tone without generic fluff."""

    return call_gemini(prompt, api_key=api_key)


def get_ai_recommendation_insights(
    dataset_summary: dict[str, Any],
    problem_type: str,
    algorithm_scores: list[dict[str, Any]],
    target_column: str,
    api_key: str | None = None,
) -> str | None:
    """Generates tailored model architecture and tuning guidance via Gemini."""
    prompt = f"""You are an enterprise MLOps Architect evaluating algorithm selection for an automated ML pipeline.
Dataset Profile:
- Target Variable: '{target_column}'
- Problem Type: {problem_type}
- Dimensions: {dataset_summary.get('rows')} rows, {dataset_summary.get('columns')} features
- High Cardinality Features: {dataset_summary.get('high_cardinality_cols', [])}
- Target Imbalance / Skew: {dataset_summary.get('imbalance_ratio', 1.0)}:1

Algorithm Suitability Rankings:
{json.dumps([{ 'algo': a.get('algorithm'), 'score': a.get('score'), 'reasons': a.get('reasons') } for a in algorithm_scores])}

Provide a deep, strategic MLOps recommendation (2-3 concise paragraphs):
1. Architectural Trade-off Analysis: Why the top-ranked model fits the feature-to-sample ratio and variance profile.
2. Hyperparameter Tuning Recommendations: Exact parameters to prioritize (e.g. n_estimators, max_depth, learning_rate, regularization).
3. Production Latency & Deployment Consideration: Model size vs inference speed in production."""

    return call_gemini(prompt, api_key=api_key)


def get_ai_shap_business_translation(
    model_name: str,
    algorithm: str,
    top_features: list[dict[str, Any]],
    local_sample: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> str | None:
    """Translates SHAP game-theoretic Shapley values into actionable business strategies via Gemini."""
    prompt = f"""You are a Principal AI Strategist and Decision Scientist.
A production model '{model_name}' ({algorithm}) was analyzed using SHAP (Shapley Additive Explanations).
Top Feature Attributions:
{json.dumps(top_features[:6])}

Sample Instance Values:
{json.dumps(local_sample) if local_sample else 'Global attribution cohort'}

Provide an actionable Business Translation (3 concise points):
1. Key Driver Synthesis: What real-world phenomena are driving these model decisions.
2. Actionable Business Levers: Concrete interventions operations/management can take on mutable features.
3. Governance & Risk Guardrail: Any spurious correlations or fairness considerations to monitor.
Keep it strictly grounded in the provided feature attributions."""

    return call_gemini(prompt, api_key=api_key)
