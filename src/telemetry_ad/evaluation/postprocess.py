def classify_event_type(run_length: int, min_collective: int = 3) -> str:
    if run_length <= 1:
        return "point"
    if run_length >= min_collective:
        return "collective"
    return "contextual"


def extract_events(binary_labels, timestamps=None, min_collective: int = 3) -> list[dict]:
    events = []
    start = None
    values = list(binary_labels)
    for i, v in enumerate(values):
        if int(v) == 1 and start is None:
            start = i
        if int(v) == 0 and start is not None:
            end = i - 1
            length = end - start + 1
            events.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "length": length,
                    "type": classify_event_type(length, min_collective=min_collective),
                    "start_ts": None if timestamps is None else str(timestamps[start]),
                    "end_ts": None if timestamps is None else str(timestamps[end]),
                }
            )
            start = None
    if start is not None:
        end = len(values) - 1
        length = end - start + 1
        events.append(
            {
                "start_idx": start,
                "end_idx": end,
                "length": length,
                "type": classify_event_type(length, min_collective=min_collective),
                "start_ts": None if timestamps is None else str(timestamps[start]),
                "end_ts": None if timestamps is None else str(timestamps[end]),
            }
        )
    return events


def event_type_counts(events: list[dict]) -> dict:
    counts = {"point": 0, "contextual": 0, "collective": 0}
    for event in events:
        event_type = event.get("type", "point")
        counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def event_overlap_metrics(true_events: list[dict], pred_events: list[dict]) -> dict:
    if not true_events and not pred_events:
        return {"event_precision": None, "event_recall": None, "event_f1": None}
    if not true_events:
        return {"event_precision": 0.0, "event_recall": None, "event_f1": None}
    if not pred_events:
        return {"event_precision": 0.0, "event_recall": 0.0, "event_f1": 0.0}

    def overlaps(a: dict, b: dict) -> bool:
        return not (a["end_idx"] < b["start_idx"] or b["end_idx"] < a["start_idx"])

    pred_hits = 0
    for pe in pred_events:
        if any(overlaps(pe, te) for te in true_events):
            pred_hits += 1

    true_hits = 0
    for te in true_events:
        if any(overlaps(te, pe) for pe in pred_events):
            true_hits += 1

    precision = pred_hits / len(pred_events) if pred_events else 0.0
    recall = true_hits / len(true_events) if true_events else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"event_precision": precision, "event_recall": recall, "event_f1": f1}
