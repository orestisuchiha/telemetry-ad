from __future__ import annotations

from pathlib import Path


def interpret_event(
    event: dict,
    *,
    total_points: int,
    drift_min_length: int = 120,
    drift_length_ratio: float = 0.25,
) -> dict:
    event_type = str(event.get("type", "point"))
    length = int(event.get("length", 0))
    end_idx = int(event.get("end_idx", 0))
    total = max(int(total_points), 1)
    ratio = length / total
    reaches_end = end_idx >= (total - 1)

    if event_type == "point":
        interpretation = "possible_fault_or_short_transient"
        rationale = "Single-window anomaly is most consistent with a short transient or isolated fault-like spike."
    elif event_type == "contextual":
        interpretation = "possible_context_dependent_abnormality"
        rationale = "Short contextual anomaly suggests behavior that is unusual relative to nearby context and could resemble attack-like or mode-specific activity."
    elif reaches_end and (length >= drift_min_length or ratio >= drift_length_ratio):
        interpretation = "possible_concept_drift_or_setpoint_shift"
        rationale = "Long collective anomaly that persists to the end of the evaluated horizon is treated as drift-like or setpoint-shift-like behavior."
    else:
        interpretation = "possible_performance_degradation_or_sustained_fault"
        rationale = "Collective anomaly over multiple consecutive windows is most consistent with degradation or a sustained fault condition."

    return {
        **event,
        "operational_interpretation": interpretation,
        "interpretation_rationale": rationale,
        "interpretation_is_heuristic": True,
        "length_ratio": ratio,
    }


def interpret_events(
    events: list[dict],
    *,
    total_points: int,
    drift_min_length: int = 120,
    drift_length_ratio: float = 0.25,
) -> list[dict]:
    return [
        interpret_event(
            event,
            total_points=total_points,
            drift_min_length=drift_min_length,
            drift_length_ratio=drift_length_ratio,
        )
        for event in events
    ]


def interpretation_counts(events: list[dict]) -> dict:
    counts = {
        "possible_fault_or_short_transient": 0,
        "possible_context_dependent_abnormality": 0,
        "possible_performance_degradation_or_sustained_fault": 0,
        "possible_concept_drift_or_setpoint_shift": 0,
    }
    for event in events:
        label = str(event.get("operational_interpretation", ""))
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def build_operational_interpretation_summary(interpreted_by_model: dict) -> tuple[dict, str]:
    summary = {"heuristic_only": True, "models": {}}
    lines = [
        "# Operational Interpretation Summary",
        "",
        "These labels are heuristic only. They are intended to support report discussion and are not causal diagnoses.",
        "",
    ]
    for model_name, events in interpreted_by_model.items():
        counts = interpretation_counts(events)
        summary["models"][model_name] = counts
        lines.append(f"## {model_name}")
        for label, count in counts.items():
            lines.append(f"- `{label}`: {count}")
        lines.append("")
    return summary, "\n".join(lines)


def save_operational_interpretation_summary(report_dir: Path, interpreted_by_model: dict) -> dict:
    summary, markdown = build_operational_interpretation_summary(interpreted_by_model)
    (Path(report_dir) / "operational_interpretation_summary.md").write_text(markdown, encoding="utf-8")
    return summary
