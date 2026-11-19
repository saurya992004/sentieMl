from typing import Any


def rank_models(
    evaluations: list[dict[str, Any]],
    problem_type: str = "classification",
    primary_metric: str = "f1",
    custom_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Ranks evaluated models using multi-objective scoring combining predictive metrics,

    inference latency, training time, and artifact size.
    """
    if not evaluations:
        return []

    # Default weights depending on problem type & primary metric
    weights = {
        primary_metric: 0.50,
        "latency": 0.20,
        "training_time": 0.15,
        "model_size": 0.15,
    }
    if custom_weights:
        weights.update(custom_weights)

    # Normalize weights
    total_w = sum(weights.values()) or 1.0
    norm_weights = {k: v / total_w for k, v in weights.items()}

    # Extract bounds for normalization
    scores = []
    latencies = [e.get("inference_latency_ms", 1.0) for e in evaluations]
    train_times = [e.get("training_time_seconds", 1.0) for e in evaluations]
    sizes = [e.get("model_size_kb", 1.0) for e in evaluations]

    min_lat, max_lat = min(latencies), max(latencies)
    min_tt, max_tt = min(train_times), max(train_times)
    min_sz, max_sz = min(sizes), max(sizes)

    higher_is_better = primary_metric not in {"mae", "rmse"}
    metric_vals = []
    for e in evaluations:
        metrics = e.get("metrics", {})
        val = float(metrics.get(primary_metric, 0.0))
        metric_vals.append(val)

    min_m, max_m = min(metric_vals), max(metric_vals)

    ranked_items = []
    for i, e in enumerate(evaluations):
        m_val = metric_vals[i]
        lat = latencies[i]
        tt = train_times[i]
        sz = sizes[i]

        # Metric score [0, 1]
        if max_m == min_m:
            s_metric = 1.0
        elif higher_is_better:
            s_metric = (m_val - min_m) / (max_m - min_m)
        else:
            s_metric = (max_m - m_val) / (max_m - min_m)

        # Latency score (lower is better) [0, 1]
        s_lat = 1.0 if max_lat == min_lat else (max_lat - lat) / (max_lat - min_lat)

        # Training time score (lower is better) [0, 1]
        s_tt = 1.0 if max_tt == min_tt else (max_tt - tt) / (max_tt - min_tt)

        # Model size score (lower is better) [0, 1]
        s_sz = 1.0 if max_sz == min_sz else (max_sz - sz) / (max_sz - min_sz)

        overall = (
            norm_weights.get(primary_metric, 0.5) * s_metric
            + norm_weights.get("latency", 0.2) * s_lat
            + norm_weights.get("training_time", 0.15) * s_tt
            + norm_weights.get("model_size", 0.15) * s_sz
        ) * 100.0

        algo = e.get("algorithm", "Unknown")
        reasoning = (
            f"Model '{algo}' achieved {primary_metric}={round(m_val, 4)} with {round(lat, 2)}ms latency, "
            f"{round(tt, 2)}s training time, and {round(sz, 1)}KB footprint. "
            f"Multi-objective score: {round(overall, 1)}/100."
        )

        ranked_items.append({
            "model_name": e.get("model_name", algo),
            "version": str(e.get("version", "1")),
            "algorithm": algo,
            "problem_type": e.get("problem_type", problem_type),
            "metrics": e.get("metrics", {}),
            "training_time_seconds": round(tt, 4),
            "inference_latency_ms": round(lat, 4),
            "model_size_kb": round(sz, 2),
            "overall_score": round(overall, 1),
            "selection_reasoning": reasoning,
        })

    # Sort descending by overall_score
    ranked_items.sort(key=lambda x: x["overall_score"], reverse=True)
    for rank, item in enumerate(ranked_items, start=1):
        item["rank"] = rank

    return ranked_items
